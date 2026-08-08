# Interview Prep: Handling LLM Inference Timeouts

**Question:** *"Can you describe a scaling issue you faced while building BiasScope and how you resolved it?"*

### The Context
BiasScope operates on a claim-centric architecture. When a user searches a topic, we scrape up to 50 raw news articles and pass them to an LLM (Llama 3 8B) to extract atomic claims. To ensure the user gets results in under a minute, we built this to be highly concurrent, firing off 50 HTTP requests to the HuggingFace serverless inference endpoint simultaneously using `asyncio`.

### The Problem
When we pushed this to production and experienced peak news cycles, we noticed that 10-15% of the articles were failing to process. When I checked the logs, the HuggingFace API was throwing `503 Service Unavailable` or `504 Gateway Timeout` errors. 

Because we were firing 50 massive text payloads at the exact same millisecond, the serverless endpoint was getting overwhelmed. It would process 40 of them, but drop the remaining 10. This was severely degrading our Data Quality Score (DQS) because we were losing 20% of our scraped data.

### The Solution
Instead of abandoning concurrency (which would have made the app unbearably slow), I implemented an **Exponential Backoff Strategy** with Jitter.

1. **Try/Except Wrapper:** I wrapped the async API call in a `try...except` block.
2. **Exponential Backoff:** If the API threw a 503 error, the script wouldn't instantly fail. It would wait $2^0$ (1) seconds, then try again. If it failed again, it waited $2^1$ (2) seconds, then 4 seconds, up to 3 retries.
3. **The Result:** The HuggingFace servers just needed a few seconds to clear their queue. By backing off, the 10 failed requests would retry a second later and successfully process. Our LLM timeout rate dropped from 14% to a flat 0%, and we maintained our high throughput without losing any data.
