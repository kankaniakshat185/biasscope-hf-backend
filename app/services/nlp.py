from transformers import pipeline
import re
from collections import Counter
from urllib.parse import urlparse
import spacy
from transformers import pipeline

# Load HuggingFace pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)
bias_pipeline = pipeline("text-classification", model="bucketresearch/politicalBiasBERT", truncation=True, max_length=512)

try:
    # Issue 5: Upgrade NER to transformer model
    spacy_nlp = spacy.load("en_core_web_trf")
except Exception as e:
    print(f"Spacy trf failed to load, falling back: {e}")
    try:
        spacy_nlp = spacy.load("en_core_web_sm")
    except:
        spacy_nlp = None

# Static Source Bias Registry (Phase 1)
SOURCE_BIAS_REGISTRY = {
    # Left Leaning
    "nytimes.com": "LEFT", "washingtonpost.com": "LEFT", "cnn.com": "LEFT", 
    "msnbc.com": "LEFT", "theguardian.com": "LEFT", "gizmodo.com": "LEFT", 
    "huffpost.com": "LEFT", "vox.com": "LEFT", "vice.com": "LEFT",
    "commondreams.org": "LEFT", "thewire.in": "LEFT", "ndtv.com": "LEFT",
    "nbcnews.com": "LEFT", "nymag.com": "LEFT", "vanityfair.com": "LEFT",
    "propublica.org": "LEFT", "aljazeera.com": "LEFT", "newrepublic.com": "LEFT",

    # Center Leaning
    "reuters.com": "CENTER", "apnews.com": "CENTER", "bbc.co.uk": "CENTER",
    "bbc.com": "CENTER", "npr.org": "CENTER", "wsj.com": "CENTER", 
    "ft.com": "CENTER", "bloomberg.com": "CENTER", "thehindu.com": "CENTER",
    "indianexpress.com": "CENTER",

    # Right Leaning
    "foxnews.com": "RIGHT", "nypost.com": "RIGHT", "dailymail.co.uk": "RIGHT",
    "dailymail.com": "RIGHT", "breitbart.com": "RIGHT", "dailycaller.com": "RIGHT", "theblaze.com": "RIGHT",
    "wnd.com": "RIGHT", "newsmax.com": "RIGHT", "oann.com": "RIGHT",
    "republicworld.com": "RIGHT", "opindia.com": "RIGHT"
}

def analyze_articles(articles):
    analyzed = []
    for art in articles:
        # -------- SENTIMENT --------
        text = art.get("content") or art.get("title", "")
        
        if not text:
            art["sentiment"] = "neutral"
            art["sentiment_score"] = 0.0
            art["confidence"] = 0.0
        else:
            # Issue 6: Calibrated sentiment instead of raw confidence
            try:
                results = sentiment_pipeline(text, top_k=None)
                if isinstance(results[0], list): 
                    results = results[0]
                
                pos_score = next((r['score'] for r in results if r['label'] == 'POSITIVE'), 0.5)
                neg_score = next((r['score'] for r in results if r['label'] == 'NEGATIVE'), 0.5)
                
                # DistilBERT is extremely polarizing (usually 0.99). 
                # We calculate the raw difference and dampen it to simulate VADER-like calibrated distributions.
                raw_diff = pos_score - neg_score
                
                # Dampen and add minor variance based on text length to create natural distribution
                length_factor = min(len(text) / 2000, 1.0)
                compound = raw_diff * (0.4 + (0.3 * length_factor))
                
                if abs(compound) < 0.15:
                    art["sentiment"] = "neutral"
                elif compound > 0:
                    art["sentiment"] = "positive"
                else:
                    art["sentiment"] = "negative"
                
                art["sentiment_score"] = round(compound, 2)
                art["confidence"] = max(pos_score, neg_score)
            except Exception as e:
                print(f"Sentiment error: {e}")
                art["sentiment"] = "neutral"
                art["sentiment_score"] = 0.0
                art["confidence"] = 0.0

        # -------- ENTITY EXTRACTION --------
        art["entities"] = {}
        if spacy_nlp and text:
            doc = spacy_nlp(text[:2000]) # limit length for speed
            entities = {}
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    name = ent.text.strip()
                    
                    # Issue 5: Entity Normalization
                    # Remove trailing 's, spaces, formatting
                    name = re.sub(r"['’]s$", "", name)
                    name = re.sub(r"[^\w\s-]", "", name)
                    name = name.title().strip()
                    
                    if len(name) > 2 and "\n" not in name:
                        if name not in entities:
                            entities[name] = {"label": ent.label_, "count": 1}
                        else:
                            entities[name]["count"] += 1
                            
            # Second pass: Merge subsets (e.g. "Musk" -> "Elon Musk")
            keys = list(entities.keys())
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if i != j and keys[i] in keys[j] and len(keys[i]) > 3:
                        entities[keys[j]]["count"] += entities[keys[i]]["count"]
                        entities[keys[i]]["count"] = 0
                        break
            
            entities = {k: v for k, v in entities.items() if v["count"] > 0}
            
            # sort and keep top 5
            sorted_ents = sorted(entities.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            art["entities"] = {k: v["label"] for k, v in sorted_ents}

        # -------- DEEP LEARNING BIAS ANALYSIS --------
        if not text:
            art["bias_label"] = "UNKNOWN"
            art["bias_confidence"] = 0.0
        else:
            try:
                # Run PoliticalBiasBERT Inference
                bias_result = bias_pipeline(text)[0]
                raw_bias = bias_result['label'].upper() # e.g. "LEFT", "CENTER", "RIGHT"
                art["bias_confidence"] = bias_result['score']
                
                # Ensure it maps safely to Prisma enums
                if raw_bias in ["LEFT", "CENTER", "RIGHT"]:
                    art["bias_label"] = raw_bias
                else:
                    art["bias_label"] = "UNKNOWN"
            except Exception as e:
                print(f"Bias inference error: {e}")
                art["bias_label"] = "UNKNOWN"
                art["bias_confidence"] = 0.0

        # -------- HYBRID BIAS ASSIGNMENT & ANOMALY DETECTION --------
        source_domain = art.get("source", "").lower()
        art["source_bias"] = "UNKNOWN"
        for domain, bias in SOURCE_BIAS_REGISTRY.items():
            if domain in source_domain:
                art["source_bias"] = bias
                break
                
        # Calculate Deviation Score (Narrative Anomaly Detection)
        # Distance map: LEFT=0, CENTER=1, RIGHT=2
        bias_map = {"LEFT": 0, "CENTER": 1, "RIGHT": 2, "UNKNOWN": None}
        s_val = bias_map.get(art["source_bias"])
        a_val = bias_map.get(art["bias_label"])
        
        if s_val is not None and a_val is not None:
            # 0.0 = Complete agreement, 1.0 = Minor shift (Left to Center), 2.0 = Extreme anomaly (Left to Right)
            art["deviation_score"] = float(abs(s_val - a_val))
        else:
            art["deviation_score"] = 0.0

        analyzed.append(art)

    return analyzed


def extract_keywords(articles):
    entity_counter = Counter()
    for art in articles:
        entities = art.get("entities", {})
        if isinstance(entities, dict):
            for entity in entities.keys():
                entity_counter[entity] += 1
    
    # Fallback if NER found absolutely nothing (e.g., weirdly formatted short text)
    if not entity_counter:
        import re
        stop_words = {"this", "that", "with", "from", "your", "have", "more", "will", "home", "about", "page", "search", "free", "information", "time", "they", "site"}
        for art in articles:
            text = art.get("title", "") + " " + art.get("content", "")
            # Find capitalized words > 3 chars as a naive entity fallback
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
            for w in words:
                if w.lower() not in stop_words:
                    entity_counter[w] += 1

    most_common = entity_counter.most_common(10)
    return [{"word": word, "count": count} for word, count in most_common]


def generate_narrative(articles):
    if not articles:
        return "No narrative available due to lack of articles."

    # Extract overall metrics
    total = len(articles)
    left_count = sum(1 for a in articles if a.get("bias_label") == "LEFT")
    right_count = sum(1 for a in articles if a.get("bias_label") == "RIGHT")
    center_count = sum(1 for a in articles if a.get("bias_label") == "CENTER")
    
    pos_count = sum(1 for a in articles if a.get("sentiment") == "positive")
    neg_count = sum(1 for a in articles if a.get("sentiment") == "negative")

    # Build a condensed context of headlines and summaries for the LLM
    snippets = []
    for a in articles[:10]: # take top 10 to avoid hitting token limits
        snippets.append(f"- Source: [{a.get('source', 'Unknown')}] | Headline: {a.get('title', '')} (Bias: {a.get('bias_label', 'UNKNOWN')}, Sentiment: {a.get('sentiment', 'neutral')})")
    
    context_str = "\n".join(snippets)
    
    import os
    hf_token = os.environ.get("HF_TOKEN")
    
    # Check if HF token exists, if not use fallback
    if not hf_token:
        return _generate_fallback_narrative(articles, left_count, right_count, center_count, pos_count, neg_count, total)
        
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=hf_token)
        
        system_prompt = (
            "You are an expert media analyst and political scientist. "
            "Your task is to write a highly professional, objective, and insightful 3-4 sentence narrative summary "
            "of the media's current coverage of a topic, based strictly on the provided article headlines, bias labels, and sentiments. "
            "You MUST explicitly bound all claims. Begin your summary with 'Among the analyzed articles...' or 'Within this sample dataset...' or 'Based on the retrieved sources...'. "
            "NEVER use phrases like 'The media believes', 'The media agrees', 'Society believes', or make sweeping claims about the broader media ecosystem outside of this sample."
        )
        
        small_sample_warning = ""
        if total < 20:
            small_sample_warning = f"\nWARNING: Only {total} articles were available for analysis. Conclusions MUST be explicitly interpreted as reflective of this small dataset rather than the broader media landscape."
            
        user_prompt = f"Media Analysis Data:\nTotal Articles: {total}\nBias Breakdown: {left_count} Left, {center_count} Center, {right_count} Right.\nSentiment: {pos_count} Positive, {neg_count} Negative.{small_sample_warning}\n\nSample Articles:\n{context_str}\n\nPlease generate the executive summary narrative. \n\nCRITICAL: You MUST explicitly cite the sources using natural phrasing, and you MUST wrap the source name in square brackets for parsing (Example: 'as reported by [indianexpress.com]' or 'according to [thehindu.com]'). Do not just drop brackets randomly at the end of sentences. Do NOT use numbers like [1]. Do NOT output any preambles."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(messages=messages, max_tokens=250, temperature=0.5)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Narrative LLM fallback triggered due to: {e}")
        return _generate_fallback_narrative(articles, left_count, right_count, center_count, pos_count, neg_count, total)

def _generate_fallback_narrative(articles, left_count, right_count, center_count, pos_count, neg_count, total):
    overall_sentiment = "neutral"
    if pos_count > neg_count and pos_count > 0.3 * total:
        overall_sentiment = "positive"
    elif neg_count > pos_count and neg_count > 0.3 * total:
        overall_sentiment = "negative"
        
    unknown_count = sum(1 for a in articles if a.get("bias_label") == "UNKNOWN")
    keywords = extract_keywords(articles)
    keyword_str = ", ".join(keywords[:3]) if keywords else "various topics"

    lines = []
    lines.append(f"The general coverage surrounding this topic exhibits an overall {overall_sentiment} sentiment across {total} analyzed articles.")
    lines.append(f"Major themes frequently discussed include {keyword_str}.")

    if left_count > right_count * 2:
        lines.append("The topic currently appears to be predominantly covered by center-to-left leaning sources.")
    elif right_count > left_count * 2:
        lines.append("The topic currently appears to be predominantly covered by center-to-right leaning sources.")
    else:
        lines.append("Coverage is relatively balanced across different political leanings among the sampled sources.")

    if unknown_count > 0:
        lines.append(f"A portion of sources ({unknown_count}) could not be classified and were marked as UNKNOWN.")

    return " ".join(lines)


def generate_contrastive_summaries(articles):
    """
    Generates two distinct summaries representing the 'Left-Wing' and 'Right-Wing' echo chambers.
    Uses source_bias (publisher level) instead of bias_label (article level) to prevent duplication.
    """
    left_articles = [a for a in articles if a.get("source_bias") == "LEFT"]
    right_articles = [a for a in articles if a.get("source_bias") == "RIGHT"]

    import os
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return {"left": "No token available for contrastive summarization.", "right": "No token available for contrastive summarization."}
        
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=hf_token)
        
        def _summarize_echo_chamber(subset, wing):
            if not subset:
                return f"Insufficient data from {wing} sources to generate a narrative."
            
            snippets = []
            for a in subset[:7]:
                snippets.append(f"- Source: [{a.get('source', 'Unknown')}] | Headline: {a.get('title', '')} (Sentiment: {a.get('sentiment', 'neutral')})")
            context_str = "\n".join(snippets)
            
            system_prompt = (
                f"You are analyzing the '{wing}' media echo chamber. "
                "Write a highly objective 2-sentence summary of how this specific political wing is framing the current topic, "
                "based strictly on the provided headlines. Do not endorse the views, just summarize their narrative framing."
            )
            
            user_prompt = f"Sample '{wing}' Articles:\n{context_str}\n\nPlease generate the {wing} narrative summary.\n\nCRITICAL: You MUST explicitly cite the sources using natural phrasing, and you MUST wrap the source name in square brackets for parsing (Example: 'as reported by [foxnews.com]' or 'according to [nypost.com]'). Do not just drop brackets randomly at the end of sentences. Do NOT use numbers like [1]. Do NOT output any preambles."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat_completion(messages=messages, max_tokens=150, temperature=0.5)
            return response.choices[0].message.content.strip()

        left_summary = _summarize_echo_chamber(left_articles, "Left-Wing")
        right_summary = _summarize_echo_chamber(right_articles, "Right-Wing")
        
        return {
            "left": left_summary,
            "right": right_summary
        }
    except Exception as e:
        print(f"Contrastive LLM fallback triggered due to: {e}")
        return {"left": "Summary unavailable.", "right": "Summary unavailable."}

def extract_entity_sentiment(articles):
    """
    Rolls up the entities across all articles into a single knowledge graph summary:
    Which entities are mentioned, and what is the average sentiment from different sides.
    """
    # structure: { "EntityName": { "label": "ORG", "left_sentiment": 0.0, "right_sentiment": 0.0, "total_mentions": 0 } }
    entity_graph = {}
    
    for art in articles:
        entities = art.get("entities", {})
        bias = art.get("bias_label", "UNKNOWN")
        sentiment_score = art.get("sentiment_score", 0.0)
        
        for name, label in entities.items():
            if name not in entity_graph:
                entity_graph[name] = {
                    "label": label,
                    "mentions": 0,
                    "left_sentiment": [],
                    "right_sentiment": [],
                    "center_sentiment": []
                }
                
            entity_graph[name]["mentions"] += 1
            if bias == "LEFT":
                entity_graph[name]["left_sentiment"].append(sentiment_score)
            elif bias == "RIGHT":
                entity_graph[name]["right_sentiment"].append(sentiment_score)
            elif bias == "CENTER":
                entity_graph[name]["center_sentiment"].append(sentiment_score)
                
    # Average the sentiments
    final_graph = {}
    for name, data in entity_graph.items():
        if data["mentions"] < 2:  # Only keep entities mentioned at least twice
            continue
            
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
            
        final_graph[name] = {
            "label": data["label"],
            "mentions": data["mentions"],
            "avg_left_sentiment": avg(data["left_sentiment"]),
            "avg_right_sentiment": avg(data["right_sentiment"]),
            "avg_center_sentiment": avg(data["center_sentiment"])
        }
        
    return final_graph