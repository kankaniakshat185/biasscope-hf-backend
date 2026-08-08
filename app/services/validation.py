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

    unique_sources = set()

    for art in articles:
        if not art.get("content") or len(art.get("content").strip()) < 50:
            missing_content_count += 1
        else:
            valid_articles.append(art)

        sentiment_scores.append(art.get("sentiment_score", 0.0))
        label = art.get("bias_label", "UNKNOWN")
        bias_counts[label] += 1

        if art.get("source"):
            unique_sources.add(art.get("source"))

    # Calculate Polarization Index via approximated Jensen-Shannon Divergence
    # We bin sentiment scores into 3 buckets (Neg, Neutral, Pos) to create distributions
    left_scores = [a.get("sentiment_score", 0.0) for a in articles if a.get("bias_label") == "LEFT"]
    right_scores = [a.get("sentiment_score", 0.0) for a in articles if a.get("bias_label") == "RIGHT"]

    def get_dist(scores):
        if not scores:
            return [1/3, 1/3, 1/3]
        neg = sum(1 for s in scores if s < -0.2)
        pos = sum(1 for s in scores if s > 0.2)
        neu = len(scores) - neg - pos
        return [neg/len(scores), neu/len(scores), pos/len(scores)]

    # R9: with no LEFT (or no RIGHT) articles at all, get_dist() falls back
    # to a uniform [1/3, 1/3, 1/3] "distribution" for that side — comparing
    # a real distribution against that arbitrary prior can easily land near
    # (or round to) 0.0, indistinguishable from "genuinely balanced
    # coverage" once displayed. Returning None here instead of a
    # coincidentally-reassuring number is honest about "we don't have
    # enough data to say" — Insight.polarizationScore is already a nullable
    # column, so this needs no schema change. The frontend renders "Not
    # enough data" for this case instead of a percentage (see the
    # dashboard's Polarization card).
    if not left_scores or not right_scores:
        polarization_score = None
    else:
        p = get_dist(left_scores)
        q = get_dist(right_scores)

        import math
        def kl_div(dist_p, dist_q):
            return sum(p_i * math.log(p_i / q_i) if p_i > 0 and q_i > 0 else 0 for p_i, q_i in zip(dist_p, dist_q, strict=False))

        m = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q, strict=False)]
        jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

        # Issue 10: JSD is polarization, NOT data quality
        polarization_score = min(jsd / 0.693, 1.0)

    # Compute ACTUAL data quality score
    total = len(articles)
    content_completeness = 1.0 - (missing_content_count / max(total, 1))
    source_diversity = min(len(unique_sources) / max(total, 1), 1.0)
    avg_content_length = sum(len(a.get("content", "") or "") for a in valid_articles) / max(len(valid_articles), 1)
    content_richness = min(avg_content_length / 1000, 1.0)

    dqs = (
        content_completeness * 0.40 +
        source_diversity * 0.30 +
        content_richness * 0.30
    )

    avg_sentiment = sum(sentiment_scores) / total if total > 0 else 0.0

    from app.services.nlp import extract_keywords
    top_keywords = extract_keywords(articles)

    # Dataset Metrics (Geographic & Political Diversity)
    countries = set()
    us_domains = ["nytimes.com", "washingtonpost.com", "cnn.com", "msnbc.com", "foxnews.com", "nypost.com", "reuters.com", "apnews.com", "npr.org", "wsj.com", "bloomberg.com", "breitbart.com", "newsmax.com"]
    uk_domains = ["bbc.co.uk", "bbc.com", "theguardian.com", "dailymail.co.uk", "dailymail.com", "ft.com"]
    in_domains = ["thehindu.com", "indianexpress.com", "thewire.in", "ndtv.com", "republicworld.com", "opindia.com", "timesofindia"]
    ca_domains = ["cbc.ca", "globalnews.ca"]

    for s in unique_sources:
        s_low = s.lower()
        if any(d in s_low for d in us_domains):
            countries.add("United States")
        elif any(d in s_low for d in uk_domains):
            countries.add("United Kingdom")
        elif any(d in s_low for d in in_domains):
            countries.add("India")
        elif any(d in s_low for d in ca_domains):
            countries.add("Canada")
        elif s_low.endswith(".au") or s_low.endswith(".au/"):
            countries.add("Australia")
        elif s_low.endswith(".eu") or s_low.endswith(".eu/"):
            countries.add("European Union")
        elif "aljazeera.com" in s_low:
            countries.add("Qatar")
        else:
            # Was `countries.add("United States")` — asserting a specific,
            # likely-wrong country for any source outside these ~45
            # hardcoded domains, which silently collapsed genuinely diverse
            # international sources into a fake single-country bucket and
            # understated real geographic diversity. "Unknown" is honest
            # about what this actually is: no match, not a US match. See
            # AUDIT_TASKS.md R7.
            countries.add("Unknown")

    countries_list = list(countries)

    # Diversity Quality Label
    # High = >3 sources AND >1 country AND (no single ideology > 70%)
    imbalance = {
        "LEFT": round(bias_counts["LEFT"] / total * 100, 1) if total > 0 else 0,
        "CENTER": round(bias_counts["CENTER"] / total * 100, 1) if total > 0 else 0,
        "RIGHT": round(bias_counts["RIGHT"] / total * 100, 1) if total > 0 else 0
    }

    max_ideology = max(imbalance.values()) if imbalance else 100

    if len(unique_sources) >= 5 and len(countries) >= 2 and max_ideology <= 60:
        div_label = "High Diversity"
    elif len(unique_sources) >= 3 and max_ideology <= 80:
        div_label = "Moderate Diversity"
    else:
        div_label = "Low Diversity"

    dataset_metrics = {
        "source_diversity": len(unique_sources),
        "source_diversity_ratio": len(unique_sources) / total if total > 0 else 0,
        "coverage_imbalance": imbalance,
        "geographic_diversity": {
            "count": len(countries_list),
            "countries": countries_list
        },
        "diversity_quality_label": div_label
    }

    return {
        "missing_content": missing_content_count,
        "valid_articles": len(valid_articles),
        "valid_articles_list": valid_articles, # Return only validated articles
        "data_quality_score": round(dqs, 2),
        "polarization_score": round(polarization_score, 2) if polarization_score is not None else None,
        "avg_sentiment": round(avg_sentiment, 3),
        "top_keywords": top_keywords,
        "bias_distribution": bias_counts,
        "dataset_metrics": dataset_metrics
    }
