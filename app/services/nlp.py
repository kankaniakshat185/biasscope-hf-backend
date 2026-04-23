from transformers import pipeline
import re
from collections import Counter
from urllib.parse import urlparse

# Load HuggingFace pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)

SOURCE_BIAS_MAP = {
    # LEFT
    "cnn.com": "LEFT",
    "msnbc.com": "LEFT",
    "huffpost.com": "LEFT",
    "nytimes.com": "LEFT",

    # RIGHT
    "foxnews.com": "RIGHT",
    "nypost.com": "RIGHT",
    "breitbart.com": "RIGHT",

    # CENTER
    "wsj.com": "CENTER",
    "reuters.com": "CENTER",
    "apnews.com": "CENTER",
    "bbc.com": "CENTER",

    # tech/media
    "theverge.com": "CENTER",
    "gizmodo.com": "CENTER",
    "wired.com": "CENTER",
    "techcrunch.com": "CENTER",

    # india specific
    "thequint.com": "LEFT",
    "thehindu.com": "LEFT",
    "scroll.in": "LEFT",
    "thewire.in": "LEFT",

    "timesofindia.indiatimes.com": "CENTER",
    "indianexpress.com": "CENTER",
    "ndtv.com": "CENTER",

    "opindia.com": "RIGHT",
    "zeenews.india.com": "RIGHT",
    "republicworld.com": "RIGHT",

    "theweek.in": "CENTER",
    "businesstoday.in": "CENTER", 

}

def extract_domain(art):
    # Prefer URL if available
    url = art.get("url", "")
    if url:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            domain = domain.split(":")[0]  # remove ports
            print("DOMAIN:", domain)
            return domain
        except:
            pass
    
    # fallback to source field
    return art.get("source", "").lower()


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

        # -------- BIAS (IMPROVED BUT SAFE) --------
        domain = extract_domain(art)

        source_bias = SOURCE_BIAS_MAP.get(domain, "UNKNOWN")

        # Hybrid logic (only modifies UNKNOWN or CENTER safely)
        if source_bias == "UNKNOWN":
            score = art["sentiment_score"]

            if score > 0.2:
                art["bias_label"] = "RIGHT"
            elif score < -0.2:
                art["bias_label"] = "LEFT"
            else:
                art["bias_label"] = "UNKNOWN"  # keep CENTER or UNKNOWN
        else:
            art["bias_label"] = source_bias

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
    
    pos_count = sum(1 for a in articles if a.get("sentiment") == "positive")
    neg_count = sum(1 for a in articles if a.get("sentiment") == "negative")
    total = len(articles)
    
    overall_sentiment = "neutral"

    if pos_count > neg_count and pos_count > 0.3 * total:
        overall_sentiment = "positive"
    elif neg_count > pos_count and neg_count > 0.3 * total:
        overall_sentiment = "negative"
        
    left_count = sum(1 for a in articles if a.get("bias_label") == "LEFT")
    right_count = sum(1 for a in articles if a.get("bias_label") == "RIGHT")
    center_count = sum(1 for a in articles if a.get("bias_label") == "CENTER")
    unknown_count = sum(1 for a in articles if a.get("bias_label") == "UNKNOWN")

    keywords = extract_keywords(articles)
    keyword_str = ", ".join(keywords[:3]) if keywords else "various topics"

    lines = []
    lines.append(f"The general coverage surrounding this topic exhibits an overall {overall_sentiment} sentiment across {total} analyzed articles.")
    lines.append(f"Major themes frequently discussed include {keyword_str}.")

    # Improved messaging
    if left_count > right_count * 2:
        lines.append("The topic currently appears to be predominantly covered by center-to-left leaning sources.")
    elif right_count > left_count * 2:
        lines.append("The topic currently appears to be predominantly covered by center-to-right leaning sources.")
    else:
        lines.append("Coverage is relatively balanced across different political leanings among the sampled sources.")

    # NEW: transparency insight
    if unknown_count > 0:
        lines.append(f"A portion of sources ({unknown_count}) could not be classified and were marked as UNKNOWN.")

    return " ".join(lines)