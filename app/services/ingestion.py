from newsapi import NewsApiClient
from newspaper import Article as NewspaperArticle
import os
import asyncio

newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))

async def ingest_articles(query: str, category: str):
    # Depending on category, we might append to the query or use domains filter.
    # For simplicity, we just search all articles matching query
    query_with_category = f"{query} {category}" if category else query
    try:
        response = newsapi.get_everything(
            q=query_with_category,
            language='en',
            sort_by='relevancy',
            page_size=15
        )
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

    articles_data = response.get('articles', [])
    
    # Run newspaper scraping concurrently to save time
    results = await asyncio.gather(*[scrape_article(a) for a in articles_data])
    return [r for r in results if r is not None]

async def scrape_article(article_data):
    url = article_data.get('url')
    if not url:
        return None
    
    # Strip protocol and www to get domain for source
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")
    
    scraped_content = ""
    try:
        # newspaper3k operations (can be blocking, wrapped to not halt asyncio totally though best practice is run_in_executor)
        # Using simple synchronous logic wrapped in an async function for demo purposes
        paper_art = NewspaperArticle(url, timeout=3)
        paper_art.download()
        paper_art.parse()
        scraped_content = paper_art.text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        # Fallback to description
        scraped_content = article_data.get('description', '')

    return {
        "title": article_data.get('title', ''),
        "url": url,
        "source": domain,
        "content": scraped_content,
        "published_at": article_data.get('publishedAt', None)
    }
