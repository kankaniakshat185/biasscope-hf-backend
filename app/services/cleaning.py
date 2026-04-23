from rapidfuzz import fuzz

def clean_and_deduplicate(raw_articles):
    cleaned = []
    seen_urls = set()
    seen_titles = []
    dupes_removed = 0

    for article in raw_articles:
        url = article.get("url")
        title = article.get("title", "").strip()
        
        if not title:
            dupes_removed += 1
            continue
            
        if url in seen_urls:
            dupes_removed += 1
            continue

        # Check title similarity
        is_duplicate = False
        for seen_title in seen_titles:
            if fuzz.token_set_ratio(title.lower(), seen_title.lower()) > 80:
                is_duplicate = True
                break
        
        if is_duplicate:
            dupes_removed += 1
            continue

        # Clean strings
        article["content"] = str(article.get("content", "")).replace("\x00", "")
        article["title"] = str(article.get("title", "")).replace("\x00", "")
        
        seen_urls.add(url)
        seen_titles.append(title)
        cleaned.append(article)
        
    return cleaned, dupes_removed
