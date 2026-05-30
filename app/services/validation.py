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
            
    # Calculate Polarization Index (Replaces Data Quality Score)
    # Measures the semantic/emotional divergence between LEFT and RIGHT media
    left_scores = [a.get("sentiment_score", 0.0) for a in articles if a.get("bias_label") == "LEFT"]
    right_scores = [a.get("sentiment_score", 0.0) for a in articles if a.get("bias_label") == "RIGHT"]
    
    if left_scores and right_scores:
        avg_left = sum(left_scores) / len(left_scores)
        avg_right = sum(right_scores) / len(right_scores)
        polarization = abs(avg_left - avg_right) / 2.0
    else:
        import statistics
        if len(sentiment_scores) > 1:
            polarization = min(1.0, statistics.stdev(sentiment_scores))
        else:
            polarization = 0.0
            
    # We map polarization to the data_quality_score field to avoid DB schema migration
    dqs = polarization

    total = len(articles)
    avg_sentiment = sum(sentiment_scores) / total if total > 0 else 0.0

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
