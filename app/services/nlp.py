from transformers import pipeline
import re
from collections import Counter
from urllib.parse import urlparse
import spacy

# Load HuggingFace pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)
bias_pipeline = pipeline("text-classification", model="bucketresearch/politicalBiasBERT", truncation=True, max_length=512)

try:
    spacy_nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Spacy failed to load: {e}")
    spacy_nlp = None

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
            # Run DistilBERT inference
            result = sentiment_pipeline(text)[0]
            label = result['label'] # 'POSITIVE' or 'NEGATIVE'
            score = result['score'] # 0.5 to 1.0 confidence
            
            # Map to compound score -1.0 to 1.0 format
            if label == "POSITIVE":
                compound = score
                if score < 0.6:  # Weak confidence -> neutral
                    compound = 0.0
                    art["sentiment"] = "neutral"
                else:
                    art["sentiment"] = "positive"
            else:
                compound = -score
                if score < 0.6:
                    compound = 0.0
                    art["sentiment"] = "neutral"
                else:
                    art["sentiment"] = "negative"
            
            art["sentiment_score"] = compound
            art["confidence"] = score

        # -------- ENTITY EXTRACTION --------
        art["entities"] = {}
        if spacy_nlp and text:
            doc = spacy_nlp(text[:2000]) # limit length for speed
            entities = {}
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    name = ent.text.strip().title()
                    # Basic cleanup
                    if len(name) > 2 and "\n" not in name:
                        if name not in entities:
                            entities[name] = {"label": ent.label_, "count": 1}
                        else:
                            entities[name]["count"] += 1
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

        analyzed.append(art)

    return analyzed


def extract_keywords(articles):
    text = " ".join([a.get("content", "") for a in articles if a.get("content")])
    
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())

    stopwords = {
        "which", "their", "there", "about", "would", "could",
        "should", "other", "after", "where"
    }

    filtered_words = [w for w in words if w not in stopwords]
    most_common = Counter(filtered_words).most_common(10)

    return [word for word, count in most_common]


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
            "of the media's current coverage of a topic, based strictly on the provided article headlines, bias labels, and sentiments."
        )
        
        user_prompt = f"Media Analysis Data:\nTotal Articles: {total}\nBias Breakdown: {left_count} Left, {center_count} Center, {right_count} Right.\nSentiment: {pos_count} Positive, {neg_count} Negative.\n\nSample Articles:\n{context_str}\n\nPlease generate the executive summary narrative. \n\nCRITICAL: You MUST explicitly cite the sources using natural phrasing like 'as reported by [indianexpress.com]' or 'according to [thehindu.com]'. Do not just drop brackets randomly at the end of sentences. Do NOT use numbers like [1] or [2]. Do NOT output any preambles like 'Here is a summary', just output the summary paragraph directly."
        
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
    """
    left_articles = [a for a in articles if a.get("bias_label") == "LEFT"]
    right_articles = [a for a in articles if a.get("bias_label") == "RIGHT"]

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
            
            user_prompt = f"Sample '{wing}' Articles:\n{context_str}\n\nPlease generate the {wing} narrative summary.\n\nCRITICAL: You MUST explicitly cite the sources using natural phrasing like 'as reported by [foxnews.com]' or 'according to [nypost.com]'. Do not just drop brackets randomly at the end of sentences. Do NOT use numbers like [1] or [2]. Do NOT output any preambles like 'Here is a summary', just output the summary text directly."
            
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