from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from .prisma_client import Prisma, Json
import uvicorn
from app.services.ingestion import ingest_articles
from app.services.cleaning import clean_and_deduplicate
from app.services.nlp import analyze_articles, generate_narrative
from app.services.validation import validate_articles
import os

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
    await prisma.connect()
    

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Biascope Backend"}

@app.post("/search")
async def create_search(
    query: str = Body(...), 
    category: str = Body(...), 
    userId: str = Body(None),
    domains: str = Body(None),
    exclude_domains: str = Body(None),
    fromDate: str = Body(None),
    toDate: str = Body(None)
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

    # 5. Narrative Gen
    summary = generate_narrative(analyzed_articles)

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
            "narrativeSummary": summary
        }
    )

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

    import requests
    import json
    
    # We use Hugging Face Serverless Inference API for free
    # Meta Llama 3 is lightning fast and brilliant for RAG tasks
    hf_token = os.environ.get("HF_TOKEN")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI intelligence analyst. Use the following news article to answer the user's question accurately. Do not invent information outside the article.
Article Context: 
{context[:2000]}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.3}
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse Llama-3 output
        generated_text = data[0]["generated_text"]
        # Strip the highly-verbose prompt out
        answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return {"answer": answer}
    except Exception as e:
        print(f"LLM API Error: {e}")
        # If the API returned an HTTP error, we need to extract the raw text (like 'Model is currently loading' or 'Unauthorized')
        error_details = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_details = e.response.text
        return {"answer": f"API Error Details: {error_details} - Please check your HuggingFace Token, or the model might be loading. Try asking again in 30 seconds!"}

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
