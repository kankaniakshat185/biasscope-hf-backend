# Deep Dive: `app/services/ingestion.py`

This file is the **Data Fetcher**. Its only job is to go out onto the internet and pull back raw news articles about whatever topic the user searched for.

---

## 1. The Theory: External APIs

We don't actually scrape Google News ourselves because handling proxies, captchas, and dynamic JavaScript rendering is a nightmare.

Instead, we use a third-party aggregator called **NewsAPI**. 
NewsAPI constantly crawls thousands of news sites and indexes their articles. When a user searches for "AI Regulation" in our app, `ingestion.py` makes an HTTP request to the NewsAPI server saying:
*"Give me the last 30 days of articles related to 'AI Regulation' in the United States."*

### API Limitations & Workarounds
NewsAPI returns a JSON object containing the Title, the Publisher (e.g., CNN), the URL, and a *snippet* of the content (usually just the first 200 characters).

But our LLM needs the **full article text** to extract meaningful claims.

To get the full text, we use a Python library called `trafilatura`.
`trafilatura` takes a raw URL, downloads the HTML of the webpage, strips away all the ads, navbars, footers, and javascript, and returns just the pure body text of the article.

---

## 2. The Code: Asynchronous Scraping

If NewsAPI returns 50 URLs, running `trafilatura` on them one by one would take forever (e.g., 50 URLs $\times$ 1 second = 50 seconds).

Just like in our LLM client, we use `asyncio` to fetch the full text for all 50 URLs concurrently. 

```python
async def fetch_full_text(session, url):
    # Sends an async HTTP GET request to the URL
    async with session.get(url) as response:
        html = await response.text()
        # Trafilatura strips the HTML and extracts the main text
        return trafilatura.extract(html)
```

By doing this asynchronously, fetching the full text of 50 articles only takes about 2-3 seconds total (the time it takes for the slowest website to respond).

---

## 3. Advanced Filtering

The `ingestion.py` file also handles the advanced filters provided by the frontend:
- **`domains` / `exclude_domains`:** If the user specifies `domains=cnn.com`, we inject that into the NewsAPI query so we only get CNN results.
- **`from_date` / `to_date`:** Injected into the query to filter by publication date. 

The resulting list of raw dictionaries `[{"title": "...", "content": "...", "source": "..."}]` is then passed to `cleaning.py` to be deduplicated.
