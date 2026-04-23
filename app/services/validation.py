def validate_articles(articles):
    missing_content_count = 0
    valid_articles = []
    
    sentiment_scores = []
    bias_counts = {
    "LEFT": 0,
    "CENTER": 0,
    "RIGHT": 0,
    "UNKNOWN": 0
}
    
    # We will do a basic keyword count extraction if it's missing (though NLP handled it)
    
    for art in articles:
        if not art.get("content") or len(art.get("content").strip()) < 50:
            missing_content_count += 1
            # Still valid for title sentiment, but score is reduced
        else:
            valid_articles.append(art)
            
        sentiment_scores.append(art.get("sentiment_score", 0.0))
        label = art.get("bias_label", "UNKNOWN")
        if label in bias_counts:
            bias_counts[label] += 1
        else:
            bias_counts[label]=0
            
    # Calculate Data Quality Score
    total = len(articles)
    if total == 0:
        dqs = 0.0
        avg_sentiment = 0.0
    else:
        missing_ratio = missing_content_count / total
        dqs = max(0.0, 1.0 - missing_ratio)
        avg_sentiment = sum(sentiment_scores) / total

    from app.services.nlp import extract_keywords
    top_keywords = extract_keywords(articles)

    return {
        "missing_content": missing_content_count,
        "valid_articles": len(valid_articles),
        "valid_articles_list": articles, # Return all after validation logic (some might just have titles)
        "data_quality_score": round(dqs, 2),
        "avg_sentiment": round(avg_sentiment, 3),
        "top_keywords": top_keywords,
        "bias_distribution": bias_counts
    }
