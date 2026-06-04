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
    query_clean = query.strip() if query else ""
    cat_clean = category.strip() if category and category.lower() != "all" else ""
    
    strict_query = query_clean
    broad_query = query_clean
    
    if query_clean and cat_clean:
        # If both exist, the broad query includes the category for better semantic matching
        broad_query = f"{query_clean} {cat_clean}"
    elif cat_clean and not query_clean:
        strict_query = cat_clean
        broad_query = cat_clean
    elif not query_clean and not cat_clean:
        strict_query = "news"
        broad_query = "news"
        
    if domains:
        from urllib.parse import urlparse
        domains_list = []
        for d in domains.split(','):
            d = d.strip()
            if not d.startswith('http'):
                d = 'http://' + d
            netloc = urlparse(d).netloc.replace("www.", "")
            parts = netloc.split('.')
            if len(parts) > 2:
                if parts[-2] in ['co', 'com', 'org', 'net', 'edu', 'gov', 'ac'] or len(parts[-2]) <= 2:
                    netloc = '.'.join(parts[-3:])
                else:
                    netloc = '.'.join(parts[-2:])
            domains_list.append(netloc)
        domains = ','.join(domains_list)

    # NewsAPI strictly forbids using BOTH domains and exclude_domains in the same request.
    # Therefore, if the user explicitly provided 'domains', we must drop all exclude logic.
    final_exclude = None
    if not domains:
        final_exclude = exclude_domains or "globenewswire.com,prnewswire.com,businesswire.com,yahoo.com,msn.com"

    import datetime
    if not from_date and not to_date:
        # If no dates are explicitly filtered, default to the last 7 days.
        # This prevents the backend from pulling 30-day-old articles just because they match string "relevancy"
        from_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')

    response = None
    try:
        # Phase 1: Hyper-Precision (Headline Match) + Popularity Sorting
        # 'popularity' ensures we get major network trending news (BBC, ESPN) instead of obscure bot-blogs
        response = newsapi.get_everything(
            qintitle=strict_query,
            domains=domains,
            exclude_domains=final_exclude,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='popularity',
            page_size=20
        )
        
        # If the headline-only search is too strict, fallback to body text
        if response.get('totalResults', 0) < 5:
            print(f"Only {response.get('totalResults')} headline matches found. Intelligently falling back to exact-phrase body search.")
            response = newsapi.get_everything(
                q=broad_query,
                domains=domains,
                exclude_domains=final_exclude,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='popularity',
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
        url_lower = url.lower()
        # Aggressive check for Audio/Podcast/Radio UI boilerplate loops
        if any(audio_path in url_lower for audio_path in ['/sounds/', '/programmes/', '/podcast', '/audio', '/video']):
            scraped_content = article_data.get('description', '') or article_data.get('title', '')
        else:
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
            
            # If the web-scraper violently fails and only pulls 2 sentences of UI boilerplate like "Sign in"
            # we instantly revert to the NewsAPI raw description block to ensure NLP has something to read.
            if len(scraped_content) < 150:
                fallback = article_data.get('description', '')
                if fallback and len(fallback) > len(scraped_content):
                    scraped_content = fallback
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

import trafilatura
from datetime import datetime
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io

async def extract_text_from_images(html_content: str, base_url: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        images = soup.find_all('img')
        extracted_text = []
        count = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            for img in images:
                if count >= 3: # limit to top 3 images to save time
                    break
                src = img.get('src')
                if not src:
                    continue
                if src.startswith('/'):
                    domain = "{0.scheme}://{0.netloc}".format(urlparse(base_url))
                    src = domain + src
                elif not src.startswith('http'):
                    continue
                
                try:
                    resp = await client.get(src)
                    if resp.status_code == 200:
                        image_data = io.BytesIO(resp.content)
                        img_obj = Image.open(image_data)
                        if img_obj.width > 200 and img_obj.height > 200:
                            text = pytesseract.image_to_string(img_obj)
                            if text and len(text.strip()) > 20:
                                extracted_text.append(text.strip())
                                count += 1
                except Exception as e:
                    print(f"OCR failed for image {src}: {e}")
        return "\n\n".join(extracted_text)
    except Exception as e:
        print(f"OCR processing failed: {e}")
        return ""

async def scrape_single_url(url: str):
    domain = urlparse(url).netloc.replace("www.", "")
    
    # 1. Direct Image URL Handling
    if url.lower().split('?')[0].endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    image_data = io.BytesIO(resp.content)
                    img_obj = Image.open(image_data)
                    text = pytesseract.image_to_string(img_obj)
                    if not text or len(text.strip()) < 20:
                        raise Exception("OCR found no meaningful text in the image.")
                    return {
                        "title": "Direct Image OCR Upload",
                        "url": url,
                        "source": domain,
                        "content": text.strip(),
                        "published_at": None
                    }
        except Exception as e:
            raise Exception(f"Failed to process direct image URL: {e}")

    # 2. Standard Web Scraping with Fallbacks
    downloaded = trafilatura.fetch_url(url)
    text = None
    title = None
    date_str = None
    
    if downloaded:
        metadata = trafilatura.extract_metadata(downloaded)
        text = trafilatura.extract(downloaded)
        if metadata:
            title = metadata.title
            if metadata.date:
                date_str = metadata.date

    # Fallback to newspaper3k if trafilatura failed to get enough text
    if not text or len(text) < 100:
        import newspaper
        try:
            config = newspaper.Config()
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
            config.request_timeout = 10
            
            article = newspaper.Article(url, config=config)
            article.download()
            article.parse()
            if article.text and len(article.text) > 100:
                text = article.text
                title = title or article.title
                if article.publish_date:
                    date_str = date_str or str(article.publish_date)
                if not downloaded:
                    downloaded = article.html
        except Exception as e:
            print(f"Newspaper3k fallback failed: {e}")

    # 3. OCR Fallback for Image-Heavy News
    if (not text or len(text) < 100) and downloaded:
        print("Text extraction insufficient. Attempting OCR on article images...")
        ocr_text = await extract_text_from_images(downloaded, url)
        if ocr_text:
            text = (text or "") + "\n\n" + ocr_text
            
    if not text or len(text) < 100:
        raise Exception("Could not extract enough main article text from the URL, even after OCR fallback.")
        
    if not title:
        title = "Direct URL Upload"
    
    return {
        "title": title,
        "url": url,
        "source": domain,
        "content": text,
        "published_at": date_str
    }
