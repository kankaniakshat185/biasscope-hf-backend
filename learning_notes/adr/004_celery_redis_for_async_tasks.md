# ADR 004: Asynchronous Task Orchestration (Celery + Redis)

**Date:** July 2026  
**Status:** Accepted  

## Context
A single search query on BiasScope initiates a pipeline that scrapes ~50 web pages, runs LLM extraction, computes vector embeddings, clusters the results, and calculates metrics. This pipeline takes anywhere from 30 to 90 seconds depending on network latency and LLM availability. FastAPI (our web framework) is designed for fast, non-blocking I/O. If we run this 90-second pipeline inside a FastAPI route, the HTTP request will block, potentially causing browser timeouts and starving the server of worker threads for other users.

## Alternatives Considered
1. **FastAPI `BackgroundTasks`:** A built-in feature of FastAPI that allows a function to run after the HTTP response is sent.
2. **Celery with Redis (Upstash):** A robust, distributed task queue system.

## Decision
We selected **Celery with Redis**.

## Justification
1. **Fault Tolerance:** If the FastAPI server crashes or is redeployed while a `BackgroundTasks` job is running, the job is lost forever. Celery stores the job in a persistent Redis queue. If the worker crashes, the job remains in the queue and will be picked up by the next available worker.
2. **Scalability:** `BackgroundTasks` run on the exact same server as the FastAPI web layer, competing for CPU and RAM. Celery allows us to completely decouple the workload. We can run the web server on a tiny instance, and run 5 Celery workers on dedicated heavy-compute instances, all coordinating through the Upstash Redis queue.
3. **Monitoring:** Celery provides robust tooling (like Flower) to monitor queue depth, retry failed tasks, and track job completion states, which is critical for long-running NLP pipelines.
