from newsapi import NewsApiClient
from newspaper import Article as NewspaperArticle
import os
import asyncio


async def ingest_articles(query: str, category: str, domains: str = None, exclude_domains: str = None, from_date: str = None, to_date: str = None):
    # Dynamically inject the key in case HF mounts secrets late, check both naming conventions
    api_key = os.environ.get('NEWS_API_KEY') or os.environ.get('NEWSAPI_KEY')
    newsapi = NewsApiClient(api_key=api_key)

    # Automatically stripping the "Category" word from the hard query helps tremendously.
    # When users search "Trump" and choose "Politics", searching "Trump Politics" explicitly 
    # yields 0 results on professional sites like WSJ because journalists don't use loose tags 
    # in the article content. We should just search the raw query!
    
    if domains:
        # Heavily sanitize the user input
        domains = domains.replace("https://", "").replace("http://", "").replace("www.", "").replace(" ", "")
        # If restricting to specific professional domains, strip category mapping to guarantee exact match hits.
        query_final = query
    else:
        # If searching the whole internet, append the category into the search string to cast a loosely accurate wide net!
        query_final = f"{query} {category}" if category else query

    # NewsAPI strictly forbids using BOTH domains and exclude_domains in the same request.
    # Therefore, if the user explicitly provided 'domains', we must drop all exclude logic.
    final_exclude = None
    if not domains:
        final_exclude = exclude_domains or "globenewswire.com,prnewswire.com,businesswire.com,yahoo.com,msn.com"

    try:
        response = newsapi.get_everything(
            q=query_final,
            domains=domains,
            exclude_domains=final_exclude,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=20
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
        # Heavily configure newspaper3k to bypass scraping defenses
        import newspaper
        config = newspaper.Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        config.request_timeout = 5
        config.fetch_images = False
        
        paper_art = NewspaperArticle(url, config=config)
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
