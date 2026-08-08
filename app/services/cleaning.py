from rapidfuzz import fuzz


def clean_and_deduplicate(raw_articles):
    """Filters raw_articles down to unique, titled articles.

    Returns (cleaned, removed_count). `removed_count` is the sum of THREE
    distinct rejection reasons — no title at all, an exact URL repeat, and
    a fuzzy-matched near-duplicate title — collapsed into one number. It's
    surfaced to users as Insight.duplicatesRemoved / the dashboard's Data
    Funnel card, currently labeled "Duplicates:", which overstates true
    duplication (a missing title isn't a duplicate) and doesn't say which
    reason actually applied. See AUDIT_TASKS.md R11 — splitting this into
    separate counters would need a schema/API change (this function's
    return shape, Insight's columns, and the frontend all touch it); the
    label was corrected instead as the proportionate fix for a Low-severity
    finding. Revisit with a real breakdown if this ever needs to answer
    "duplicates vs. missing titles" for real, not just avoid the wrong word.
    """
    cleaned = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    removed_count = 0

    for article in raw_articles:
        url = article.get("url")
        title = article.get("title", "").strip()

        if not title:
            removed_count += 1
            continue

        if url in seen_urls:
            removed_count += 1
            continue

        # Check title similarity
        is_duplicate = False
        for seen_title in seen_titles:
            if fuzz.token_set_ratio(title.lower(), seen_title.lower()) > 80:
                is_duplicate = True
                break

        if is_duplicate:
            removed_count += 1
            continue

        # R12: build a NEW dict rather than mutating `article` (a raw_articles
        # element) in place — this used to mean `raw_articles` and `cleaned`
        # silently shared the same dict objects, so anything that mutated an
        # article later in the pipeline (analyze_articles does) was also,
        # invisibly, mutating raw_articles. Nothing currently reads
        # raw_articles again after this call, so no bug existed today — but
        # it was a hidden coupling waiting for a future change to trip over.
        cleaned_article = {
            **article,
            "content": str(article.get("content", "")).replace("\x00", ""),
            "title": str(article.get("title", "")).replace("\x00", ""),
        }

        seen_urls.add(url)
        seen_titles.append(title)
        cleaned.append(cleaned_article)

    return cleaned, removed_count
