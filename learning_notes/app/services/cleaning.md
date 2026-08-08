# Deep Dive: `app/services/cleaning.py`

This file handles **Data Sanitization and Deduplication**. After `ingestion.py` pulls raw HTML/text from the web, `cleaning.py` ensures that the data is pristine before we spend expensive LLM tokens analyzing it.

---

## 1. Why do we need cleaning?

When you scrape the web, you get a lot of garbage:
- **Syndicated Content:** Reuters publishes an article. Yahoo News, MSN, and 50 other sites automatically republish the exact same article word-for-word. We don't want to analyze the exact same article 50 times; it skews our dataset.
- **Paywalls:** Sometimes `trafilatura` extracts text that just says: *"Please subscribe to read the full article."*
- **Formatting errors:** Bizarre Unicode characters, massive blocks of whitespace, or leftover javascript code.

## 2. Fuzzy Matching Deduplication

To solve the Syndicated Content problem, we use **Fuzzy Matching** (specifically, calculating the Levenshtein distance or SequenceMatcher ratio between two strings).

If two articles have text that is 95% identical, we throw one of them away.

### The Algorithm:
1. Create an empty list called `unique_articles`.
2. Loop through every scraped article.
3. For each article, compare it against all articles currently in `unique_articles`.
4. If the similarity ratio is $> 0.95$, it's a duplicate. Ignore it.
5. If it's $< 0.95$, it's a new unique article. Add it to `unique_articles`.

This guarantees that every article we process is a distinct journalistic piece, preventing our Polarization Scores from being skewed by mass-republished wire stories.

## 3. Heuristic Filtering

We also run the text through several heuristics (rules of thumb) to drop bad data:
- **Length Check:** If the article is less than 500 characters, it's probably a stub or a paywall notice. We drop it.
- **Stopword Ratio:** Genuine English text has a predictable ratio of "stopwords" (the, and, is, in). If an article is just a list of keywords or a broken HTML table, the stopword ratio will be very low. We drop it.
- **Regex Cleaning:** We use Regular Expressions (`re`) to strip out multiple newlines, weird whitespace, and common boilerplate phrases like "Click here to subscribe".
