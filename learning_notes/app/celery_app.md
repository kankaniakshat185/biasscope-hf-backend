# Deep Dive: `app/celery_app.py` & Background Tasks

When a user triggers a massive NLP pipeline, we can't freeze the FastAPI web server. We need a way to say, *"Hey, someone else handle this heavy math in the background."*

That "someone else" is **Celery**.

---

## 1. The Theory: Message Brokers

If FastAPI is the front-desk receptionist, Celery is the warehouse worker. 
But how does the receptionist hand a piece of paper to the warehouse worker? They need a basket to drop the paper into.

That basket is called a **Message Broker**. We use **Redis** (hosted on Upstash).

### The Flow:
1. **FastAPI (`main.py`):** Receives the HTTP request. It calls `snapshot_task.delay(query="AI Regulation")`.
2. **The Broker (Redis):** FastAPI packages that function call into a JSON message and drops it into a Redis queue.
3. **The Worker (Celery):** A separate Python process (`celery worker`) is constantly watching the Redis queue. It sees the message, picks it up, and starts executing the `snapshot_task` function.

This is a totally decoupled architecture. You could run FastAPI on a server in New York, and have 10 Celery workers running on heavy GPUs in Tokyo, all communicating instantly through Redis.

---

## 2. Deep Dive: `app/tasks/snapshot_task.py`

This file is the **Orchestrator**. It glues together all the individual `services` we wrote.

When Celery picks up a task, it executes `run_snapshot_pipeline` in this file.

### The Pipeline Steps:
1. **Ingestion:** Calls `ingest_articles(query)`. Gets raw HTML from the web.
2. **Cleaning:** Calls `clean_and_deduplicate(raw_articles)`. Drops junk and exact duplicates.
3. **Extraction & NLP:** 
   - Loops through the clean articles.
   - Calls `process_and_store_claims` (Llama-3 extraction).
   - Calls `analyze_articles` (Sentiment and Bias pipelines).
4. **Clustering:** Calls `run_claim_clustering()`. Merges all the extracted claims into unified Events using `pgvector` Cosine Similarity.
5. **Consensus & Echo Chambers:** Calls `run_event_detection()`. Calculates the cross-ideological consensus and generates the contrastive Left vs. Right summaries.
6. **Validation:** Calls `validate_articles()`. Computes the Data Quality Score and Polarization (Jensen-Shannon) score.

Once all 6 steps are complete, it updates the `Search` record in the PostgreSQL database to `status = 'COMPLETED'`. 

The frontend, which has been polling the database, sees the `COMPLETED` flag, fetches the data, and renders the dashboard!
