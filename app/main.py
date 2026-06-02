from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .prisma_client import Prisma, Json
import uvicorn
from app.services.cleaning import clean_and_deduplicate
from app.services.ingestion import ingest_articles, scrape_single_url
from app.services.nlp import analyze_articles, generate_narrative, generate_contrastive_summaries, extract_entity_sentiment
from app.services.validation import validate_articles
from app.services.extraction import process_and_store_claims
from app.services.clustering import run_claim_clustering, run_event_detection
import os
import io
from PIL import Image
import pytesseract

app = FastAPI(title="Biascope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prisma = Prisma()

@app.on_event("startup")
async def startup():
    print("Synchronizing database schema...")
    os.system("python3 -m prisma db push --skip-generate")
    await prisma.connect()
    

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Biascope Backend"}

async def background_phase2_pipeline(search_id: str):
    print(f"Starting background Phase 2 pipeline for search {search_id}...")
    try:
        search_record = await prisma.search.find_unique(where={"id": search_id})
        query = search_record.query if search_record else ""
        
        articles = await prisma.article.find_many(where={"searchId": search_id})
        for art in articles:
            if art.content:
                await process_and_store_claims(
                    prisma,
                    art.id,
                    art.content,
                    art.source,
                    art.url,
                    art.publishedAt,
                    query,
                )
        
        # Run Step 6 & 7 locally
        await run_claim_clustering(prisma)
        await run_event_detection(prisma)
        print("Phase 2 pipeline complete.")
    except Exception as e:
        print(f"Phase 2 pipeline error: {e}")

@app.post("/search")
async def create_search(
    query: str = Body(...), 
    category: str = Body(...), 
    userId: str = Body(None),
    domains: str = Body(None),
    exclude_domains: str = Body(None),
    fromDate: str = Body(None),
    toDate: str = Body(None),
    background_tasks: BackgroundTasks = None
):
    print(f"Starting search for: {query} in {category}")
    
    # 1. Ingestion
    raw_articles = await ingest_articles(
        query=query, 
        category=category, 
        domains=domains, 
        exclude_domains=exclude_domains, 
        from_date=fromDate, 
        to_date=toDate
    )
    if not raw_articles:
        raise HTTPException(
            status_code=404, 
            detail=f"NewsAPI error or No articles found for: query='{query}', domains='{domains}', fromDate='{fromDate}'. Please ensure domains are properly formatted (e.g. 'wsj.com') and dates are within the last 30 days."
        )

    # 2. Cleaning
    cleaned_articles, dupes_removed = clean_and_deduplicate(raw_articles)

    # 3. NLP
    analyzed_articles = analyze_articles(cleaned_articles)

    # 4. Validation
    validation_metrics = validate_articles(analyzed_articles)
    
    valid_articles_list = validation_metrics.get("valid_articles_list", [])
    if not valid_articles_list:
        raise HTTPException(status_code=404, detail="All scraped articles failed validation or were empty.")

    validation_metrics["total_articles"] = len(raw_articles)
    validation_metrics["duplicates_removed"] = dupes_removed

    # 5. Narrative Gen & Contrastive Echo Chambers & Entity Extraction
    summary = generate_narrative(analyzed_articles)
    contrastive_summaries = generate_contrastive_summaries(analyzed_articles)
    entity_sentiment_graph = extract_entity_sentiment(analyzed_articles)
    
    # Replace softmax average with Source Reliability Confidence
    valid_articles = validation_metrics.get("valid_articles_list", [])
    if valid_articles:
        reliability_scores = []
        high = ["reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "npr.org", "thehindu.com", "indianexpress.com", "ft.com", "wsj.com", "bloomberg.com", "theguardian.com", "nytimes.com", "washingtonpost.com"]
        mixed = ["foxnews.com", "cnn.com", "msnbc.com", "dailymail.co.uk", "dailymail.com", "nypost.com", "vice.com", "gizmodo.com"]
        low = ["breitbart.com", "infowars.com", "dailycaller.com", "wnd.com", "newsmax.com", "oann.com"]
        
        cred_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        for a in valid_articles:
            s = a.get("source", "").lower()
            if any(h in s for h in high):
                reliability_scores.append(0.95)
                cred_counts["High"] += 1
            elif any(m in s for m in mixed):
                reliability_scores.append(0.60)
                cred_counts["Medium"] += 1
            elif any(l in s for l in low):
                reliability_scores.append(0.20)
                cred_counts["Low"] += 1
            else:
                reliability_scores.append(0.50) # default unknown
                cred_counts["Unknown"] += 1
        avg_confidence = sum(reliability_scores) / len(reliability_scores)
    else:
        avg_confidence = 0.0
        cred_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        
    drift_metrics = {
        "source_reliability_confidence": avg_confidence,
        "credibility_breakdown": cred_counts
    }

    # 6. Store to DB
    search_record = await prisma.search.create(
        data={
            "query": query,
            "category": category,
            "userId": userId
        }
    )

    # Insert articles
    article_creates = []
    for art in validation_metrics["valid_articles_list"]:
        article_creates.append({
            "searchId": search_record.id,
            "title": art.get("title", "No Title"),
            "content": art.get("content", ""),
            "source": art.get("source", "Unknown"),
            "url": art.get("url", ""),
            "sentiment": art.get("sentiment", "neutral"),
            "sentimentScore": float(art.get("sentiment_score", 0.0)),
            "biasLabel": art.get("bias_label", "UNKNOWN"),
            "sourceBias": art.get("source_bias", "UNKNOWN"),
            "deviationScore": float(art.get("deviation_score", 0.0)),
            "entities": Json(art.get("entities", {})),
            "publishedAt": art.get("published_at")
        })
    
    if article_creates:
        await prisma.article.create_many(data=article_creates)

    # Insert Insights
    insight_record = await prisma.insight.create(
        data={
            "search": {"connect": {"id": search_record.id}},
            "avgSentiment": float(validation_metrics.get("avg_sentiment", 0.0)),
            "topKeywords": Json(validation_metrics.get("top_keywords", [])),
            "biasDistribution": Json(validation_metrics.get("bias_distribution", {"LEFT": 0, "CENTER": 0, "RIGHT": 0, "UNKNOWN": 0})),
            "dataQualityScore": float(validation_metrics.get("data_quality_score", 0.0)),
            "totalArticles": validation_metrics["total_articles"],
            "validArticles": validation_metrics["valid_articles"],
            "duplicatesRemoved": validation_metrics["duplicates_removed"],
            "missingContent": validation_metrics["missing_content"],
            "datasetMetrics": Json(validation_metrics.get("dataset_metrics", {})),
            "narrativeSummary": summary,
            "leftWingSummary": contrastive_summaries.get("left", ""),
            "rightWingSummary": contrastive_summaries.get("right", ""),
            "entitySentiment": Json(entity_sentiment_graph),
            "driftMetrics": Json(drift_metrics)
        }
    )

    if background_tasks:
        background_tasks.add_task(background_phase2_pipeline, search_record.id)

    return {"search_id": search_record.id, "message": "Search processed successfully."}

@app.get("/results/{search_id}")
async def get_results(search_id: str):
    search_record = await prisma.search.find_unique(
        where={"id": search_id},
        include={"articles": True, "insights": True}
    )
    if not search_record:
        raise HTTPException(status_code=404, detail="Search not found")
    
    return search_record

@app.get("/results/{search_id}/intelligence")
async def get_search_intelligence(search_id: str):
    # Fetch all evidence belonging to articles from this search
    articles = await prisma.article.find_many(where={"searchId": search_id})
    article_ids = [a.id for a in articles]
    
    if not article_ids:
        return {"events": [], "clusters": [], "claims": [], "metrics": {}}

    evidence_records = await prisma.evidence.find_many(
        where={"articleId": {"in": article_ids}},
        include={"claim": True}
    )
    
    claim_ids = list(set([e.claimId for e in evidence_records if e.claimId]))
    
    claims = await prisma.claim.find_many(
        where={"id": {"in": claim_ids}},
        include={
            "evidence": True,
            "cluster": {
                "include": {
                    "event": True
                }
            }
        }
    )
    
    # Process into clean structured output
    formatted_claims = []
    clusters_map = {}
    events_map = {}
    
    for c in claims:
        # Build claim
        fc = {
            "id": c.id,
            "canonicalClaim": c.canonicalClaim,
            "claimType": getattr(c, 'claimType', 'EVENT') or 'EVENT',
            "qualityScore": getattr(c, 'qualityScore', 0) or 0,
            "confidence": c.confidence,
            "evidenceCount": len(c.evidence),
            "sources": list(set([e.source for e in c.evidence])),
            "evidence": [{"sentence": e.sentence, "source": e.source, "publishedAt": e.publishedAt, "url": e.url} for e in c.evidence],
            "clusterId": c.clusterId
        }
        formatted_claims.append(fc)
        
        # Build cluster
        if c.cluster:
            cid = c.cluster.id
            if cid not in clusters_map:
                clusters_map[cid] = {
                    "id": cid,
                    "title": c.cluster.title,
                    "canonicalClaim": getattr(c.cluster, 'canonicalClaim', '') or c.canonicalClaim,
                    "consensusScore": getattr(c.cluster, 'consensusScore', 0) or 0,
                    "eventId": c.cluster.eventId,
                    "claims": [],
                    "evidenceCount": 0,
                    "sources": set()
                }
            if fc["canonicalClaim"] not in clusters_map[cid]["claims"]:
                clusters_map[cid]["claims"].append(fc["canonicalClaim"])
            clusters_map[cid]["evidenceCount"] += fc["evidenceCount"]
            for s in fc["sources"]:
                clusters_map[cid]["sources"].add(s)
                
            # Build event
            if c.cluster.event:
                eid = c.cluster.event.id
                if eid not in events_map:
                    events_map[eid] = {
                        "id": eid,
                        "title": c.cluster.event.title,
                        "description": getattr(c.cluster.event, 'description', '') or '',
                        "importanceScore": getattr(c.cluster.event, 'importanceScore', 0) or 0,
                        "clusters": [],
                        "claimCount": 0,
                        "evidenceCount": 0,
                        "sources": set()
                    }
                if c.cluster.title not in events_map[eid]["clusters"]:
                    events_map[eid]["clusters"].append(c.cluster.title)
                events_map[eid]["claimCount"] += 1
                events_map[eid]["evidenceCount"] += fc["evidenceCount"]
                for s in fc["sources"]:
                    events_map[eid]["sources"].add(s)
                    
    # Format maps to lists
    formatted_clusters = list(clusters_map.values())
    for cl in formatted_clusters:
        cl["sources"] = list(cl["sources"])
        cl["sourceCount"] = len(cl["sources"])
        
    formatted_events = list(events_map.values())
    for ev in formatted_events:
        ev["sources"] = list(ev["sources"])
        ev["sourceCount"] = len(ev["sources"])
        
    return {
        "metrics": {
            "articlesProcessed": len(article_ids),
            "claimsExtracted": len(evidence_records),
            "canonicalClaims": len(formatted_claims),
            "clusters": len(formatted_clusters),
            "events": len(formatted_events)
        },
        "claims": sorted(formatted_claims, key=lambda x: x["evidenceCount"], reverse=True),
        "clusters": sorted(formatted_clusters, key=lambda x: x["evidenceCount"], reverse=True),
        "events": sorted(formatted_events, key=lambda x: x.get("importanceScore", 0), reverse=True)
    }

@app.post("/chat-with-article")
async def chat_with_article(
    articleId: str = Body(...),
    message: str = Body(...)
):
    # Retrieve the specific article context
    article = await prisma.article.find_unique(where={"id": articleId})
    if not article:
        raise HTTPException(status_code=404, detail="Article not found in database.")
        
    context = article.content if article.content else article.title

    import os
    
    # We use Hugging Face Serverless Inference API for free
    hf_token = os.environ.get("HF_TOKEN")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=hf_token)
        
        messages = [
            {
                "role": "system", 
                "content": f"You are an expert AI intelligence analyst. Use the following news article to answer the user's question accurately. Do not invent information outside the article. If the user asks something unrelated to the article, politely decline. \n\nArticle Context: \n{context[:2500]}"
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        response = client.chat_completion(messages=messages, max_tokens=250, temperature=0.3)
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"answer": f"API Error Details: {str(e)} - Please check your HuggingFace Token! (If you get a 403, you may need to accept the Llama-3 terms at huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)"}

@app.post("/chat-with-summary")
async def chat_with_summary(
    searchId: str = Body(...),
    message: str = Body(...)
):
    insight = await prisma.insight.find_first(where={"searchId": searchId})
    if not insight:
        raise HTTPException(status_code=404, detail="Summary not found in database.")
        
    context = insight.narrativeSummary

    import os
    hf_token = os.environ.get("HF_TOKEN")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=hf_token)
        
        messages = [
            {
                "role": "system", 
                "content": f"You are an expert AI intelligence analyst. Use the following overarching narrative summary to answer the user's question accurately. Do not invent information outside the summary.\n\nSummary Context: \n{context}"
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        response = client.chat_completion(messages=messages, max_tokens=250, temperature=0.3)
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"answer": f"API Error Details: {str(e)}"}

@app.post("/analyze-url")
async def analyze_url_endpoint(
    url: str = Body(...),
    userId: str = Body(None)
):
    print(f"Starting single URL analysis for: {url}")
    
    # 1. Ingestion
    try:
        raw_article = await scrape_single_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract article from URL: {str(e)}")

    if not raw_article:
        raise HTTPException(status_code=400, detail="Could not extract content from the provided URL.")

    # 2. NLP (Since it's 1 article, no cleaning needed)
    analyzed_articles = analyze_articles([raw_article])

    # 3. Validation
    validation_metrics = validate_articles(analyzed_articles)
    
    valid_articles_list = validation_metrics.get("valid_articles_list", [])
    if not valid_articles_list:
        raise HTTPException(status_code=400, detail="The extracted text was too short or invalid for AI analysis.")

    validation_metrics["total_articles"] = 1
    validation_metrics["duplicates_removed"] = 0

    # 4. Intelligence
    summary = generate_narrative(analyzed_articles)
    entity_sentiment_graph = extract_entity_sentiment(analyzed_articles)
    
    # Calculate model drift metrics
    s = valid_articles_list[0].get("source", "").lower()
    high = ["reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "npr.org", "thehindu.com", "indianexpress.com", "ft.com", "wsj.com", "bloomberg.com", "theguardian.com", "nytimes.com", "washingtonpost.com"]
    mixed = ["foxnews.com", "cnn.com", "msnbc.com", "dailymail.co.uk", "dailymail.com", "nypost.com", "vice.com", "gizmodo.com"]
    low = ["breitbart.com", "infowars.com", "dailycaller.com", "wnd.com", "newsmax.com", "oann.com"]
    
    if any(h in s for h in high): 
        conf = 0.95
        cat = "High"
    elif any(m in s for m in mixed): 
        conf = 0.60
        cat = "Medium"
    elif any(l in s for l in low): 
        conf = 0.20
        cat = "Low"
    else: 
        conf = 0.50
        cat = "Unknown"
    
    drift_metrics = {
        "source_reliability_confidence": conf,
        "credibility_breakdown": {"High": int(cat=="High"), "Medium": int(cat=="Medium"), "Low": int(cat=="Low"), "Unknown": int(cat=="Unknown")}
    }

    # 5. Store to DB
    search_record = await prisma.search.create(
        data={
            "query": url,
            "category": "Single URL",
            "userId": userId
        }
    )

    # Insert article
    art = valid_articles_list[0]

    # Format date for Prisma
    published_at = art.get("published_at")
    if published_at and len(published_at) == 10:  # e.g., 'YYYY-MM-DD'
        published_at = f"{published_at}T00:00:00Z"
    await prisma.article.create(
        data={
            "search": {"connect": {"id": search_record.id}},
            "title": art.get("title", "No Title"),
            "content": art.get("content", ""),
            "source": art.get("source", "Unknown"),
            "url": art.get("url", ""),
            "sentiment": art.get("sentiment", "neutral"),
            "sentimentScore": float(art.get("sentiment_score", 0.0)),
            "biasLabel": art.get("bias_label", "UNKNOWN"),
            "sourceBias": art.get("source_bias", "UNKNOWN"),
            "deviationScore": float(art.get("deviation_score", 0.0)),
            "entities": Json(art.get("entities", {})),
            "publishedAt": published_at
        }
    )
    
    # Insight logic specifically ignores echo chambers for single URL
    insight_record = await prisma.insight.create(
        data={
            "search": {"connect": {"id": search_record.id}},
            "avgSentiment": float(validation_metrics.get("avg_sentiment", 0.0)),
            "topKeywords": Json(validation_metrics.get("top_keywords", [])),
            "biasDistribution": Json(validation_metrics.get("bias_distribution", {"LEFT": 0, "CENTER": 0, "RIGHT": 0, "UNKNOWN": 0})),
            "dataQualityScore": float(validation_metrics.get("data_quality_score", 0.0)),
            "totalArticles": 1,
            "validArticles": 1,
            "duplicatesRemoved": 0,
            "missingContent": 0,
            "narrativeSummary": summary,
            "leftWingSummary": "",
            "rightWingSummary": "",
            "entitySentiment": Json(entity_sentiment_graph),
            "driftMetrics": Json(drift_metrics)
        }
    )

    return {"search_id": search_record.id, "message": "URL processed successfully."}

@app.post("/analyze-upload")
async def analyze_upload_endpoint(
    file: UploadFile = File(...),
    userId: str = Form(None)
):
    print(f"Starting single image upload analysis for: {file.filename}")
    
    # 1. Image OCR Ingestion
    content = await file.read()
    image_data = io.BytesIO(content)
    try:
        img_obj = Image.open(image_data)
        text = pytesseract.image_to_string(img_obj)
        if not text or len(text.strip()) < 20:
            raise Exception("OCR found no meaningful text in the uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image OCR: {str(e)}")

    extracted_title = f"Local Upload: {file.filename}"
    extracted_source = "Local System"
    
    try:
        from huggingface_hub import InferenceClient
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
            prompt = f"You are a helpful assistant. I have extracted raw OCR text from an image/infographic. Identify the most likely Title of the article/image and the Publisher/Source if it's visible. If not, make a highly concise, educated guess based on the text. Return strictly in this format:\nTitle: <title>\nSource: <source>\n\nOCR Text:\n{text[:1500]}"
            res = client.chat_completion(messages=[{"role":"user", "content":prompt}], max_tokens=100, temperature=0.2)
            lines = res.choices[0].message.content.strip().split('\n')
            for line in lines:
                if line.lower().startswith("title:"):
                    extracted_title = line.split(":", 1)[1].strip().replace('"', '')
                elif line.lower().startswith("source:"):
                    extracted_source = line.split(":", 1)[1].strip().replace('"', '')
    except Exception as e:
        print(f"Failed to dynamically extract OCR metadata: {e}")

    raw_article = {
        "title": extracted_title,
        "url": "local_upload",
        "source": extracted_source,
        "content": text.strip(),
        "published_at": None
    }

    # 2. NLP
    analyzed_articles = analyze_articles([raw_article])

    # 3. Validation
    validation_metrics = validate_articles(analyzed_articles)
    
    valid_articles_list = validation_metrics.get("valid_articles_list", [])
    if not valid_articles_list:
        raise HTTPException(status_code=400, detail="The extracted text was too short or invalid for AI analysis.")

    validation_metrics["total_articles"] = 1
    validation_metrics["duplicates_removed"] = 0

    # 4. Intelligence
    summary = generate_narrative(analyzed_articles)
    entity_sentiment_graph = extract_entity_sentiment(analyzed_articles)
    
    # Model Drift Metrics (Source Reliability Confidence)
    reliability_scores = []
    high = ["reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "npr.org", "thehindu.com", "indianexpress.com", "ft.com", "wsj.com", "bloomberg.com", "theguardian.com", "nytimes.com", "washingtonpost.com"]
    mixed = ["foxnews.com", "cnn.com", "msnbc.com", "dailymail.co.uk", "dailymail.com", "nypost.com", "vice.com", "gizmodo.com"]
    low = ["breitbart.com", "infowars.com", "dailycaller.com", "wnd.com", "newsmax.com", "oann.com"]
    
    cred_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for a in analyzed_articles:
        s = a.get("source", "").lower()
        if any(h in s for h in high): 
            reliability_scores.append(0.95)
            cred_counts["High"] += 1
        elif any(m in s for m in mixed): 
            reliability_scores.append(0.60)
            cred_counts["Medium"] += 1
        elif any(l in s for l in low): 
            reliability_scores.append(0.20)
            cred_counts["Low"] += 1
        else: 
            reliability_scores.append(0.50)
            cred_counts["Unknown"] += 1
        
    avg_conf = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.0
    
    drift_metrics = {
        "source_reliability_confidence": avg_conf,
        "credibility_breakdown": cred_counts
    }

    # 5. DB Persistence
    from datetime import datetime
    iso_date = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

    search_record = await prisma.search.create(
        data={
            "query": f"Image Upload: {file.filename}",
            "category": "Single URL",
            "userId": userId,
            "articles": {
                "create": [
                    {
                        "title": a.get("title", "Unknown"),
                        "url": a.get("url", "unknown"),
                        "source": a.get("source", "Unknown"),
                        "publishedAt": a.get("published_at") or iso_date,
                        "sentiment": a.get("sentiment", "neutral"),
                        "sentimentScore": float(a.get("sentiment_score", 0.0)),
                        "biasLabel": a.get("bias_label", "UNKNOWN"),
                        "sourceBias": a.get("source_bias", "UNKNOWN"),
                        "deviationScore": float(a.get("deviation_score", 0.0)),
                        "entities": Json(a.get("entities", {}))
                    } for a in analyzed_articles
                ]
            }
        }
    )

    insight_record = await prisma.insight.create(
        data={
            "search": {"connect": {"id": search_record.id}},
            "avgSentiment": float(validation_metrics.get("avg_sentiment", 0.0)),
            "topKeywords": Json(validation_metrics.get("top_keywords", [])),
            "biasDistribution": Json(validation_metrics.get("bias_distribution", {"LEFT": 0, "CENTER": 0, "RIGHT": 0, "UNKNOWN": 0})),
            "dataQualityScore": float(validation_metrics.get("data_quality_score", 0.0)),
            "totalArticles": 1,
            "validArticles": 1,
            "duplicatesRemoved": 0,
            "missingContent": 0,
            "datasetMetrics": Json(validation_metrics.get("dataset_metrics", {})),
            "narrativeSummary": summary,
            "leftWingSummary": "",
            "rightWingSummary": "",
            "entitySentiment": Json(entity_sentiment_graph),
            "driftMetrics": Json(drift_metrics)
        }
    )

    return {"search_id": search_record.id, "message": "File processed successfully."}

@app.get("/history")
async def get_history(userId: str = None):
    # Retrieve past searches
    queries = {}
    if userId:
        queries = {"where": {"userId": userId}}
        
    searches = await prisma.search.find_many(
        **queries,
        order={"createdAt": "desc"},
        include={"insights": True}
    )
    return searches

@app.delete("/history/{search_id}")
async def delete_search(search_id: str):
    search_record = await prisma.search.find_unique(where={"id": search_id})
    if not search_record:
        raise HTTPException(status_code=404, detail="Search not found")
        
    await prisma.search.delete(where={"id": search_id})
    return {"message": "Search and all associated data permanently deleted."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
