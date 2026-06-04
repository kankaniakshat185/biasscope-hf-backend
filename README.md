# 🏛️ BiasScope Core Engine (Backend)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0+-316192.svg?logo=postgresql)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BiasScope Core Engine is the high-performance, claim-centric natural language processing backend that powers the BiasScope Intelligence Dashboard. Built specifically to transcend basic keyword matching, this backend consumes raw media coverage and distills it into rigorous, cross-ideological, evidence-backed events.

## 🚀 Key Differentiators & Architecture

Traditional sentiment engines rely on article-level metrics. BiasScope's engine uses a **Claim-Centric** pipeline. It extracts discrete claims from the news, links them to evidence (the exact sentences), and runs clustering and consensus algorithms across the entire database to find exactly where media outlets agree and disagree.

### High-Level Pipeline Architecture

```mermaid
graph TD
    A[Raw Article Ingestion] --> B[Data Cleaning & Deduplication]
    B --> C[LLM Claim Extraction Engine]
    C --> D[(Global Claim Database w/ pgvector)]
    D --> E[Claim Clustering via Cosine Similarity]
    E --> F[Event & Narrative Generation]
    F --> G[Cross-Ideological Consensus Calculation]
    G --> H[FastAPI Results Endpoint]
```

## 🛠️ Tech Stack
- **Framework:** FastAPI (High-performance ASGI framework)
- **Database:** PostgreSQL (with `pgvector` for semantic search)
- **ORM:** Prisma Client Python
- **NLP / ML:** `sentence-transformers/all-MiniLM-L6-v2`, HuggingFace Inference API (Meta-Llama-3-8B-Instruct)
- **Background Tasks:** Celery + Upstash Redis (For asynchronous polling and snapshot generation)
- **Hosting:** HuggingFace Spaces / Render

## ✨ Core Features

*   **🔍 Multi-Phase Media Ingestion**
    *   Ingests articles using NewsAPI, falling back gracefully across query modes to ensure hyper-relevant content retrieval.
*   **🧬 LLM Claim Extraction & Normalization**
    *   Breaks down raw articles into verifiable, factual claims utilizing Llama 3 Instruct models, storing raw evidence for citations.
*   **🌐 Semantic Claim Clustering (`pgvector`)**
    *   Embeds all extracted claims using SentenceTransformers. Uses cosine similarity clustering to merge semantically identical claims into single "Canonical/Core Claims" across multiple sources.
*   **⚖️ Contrastive Echo Chambers (`BETA`)**
    *   Leverages advanced LLM prompting to isolate and analyze how the "Left-Wing" vs "Right-Wing" media ecosystems are rhetorically framing the exact same event.
*   **🤝 Cross-Ideological Consensus Engine**
    *   Programmatically calculates a `consensusScore` based on the publisher diversity supporting a single claim. Claims consistently reported across partisan lines are tagged automatically.
*   **📊 Entity Sentiment Graphing (`BETA`)**
    *   Rolls up Named Entity Recognition (NER) tags into a structured graph, calculating the exact sentiment polarization per entity across all analyzed articles.
*   **⏳ Automated Topic Snapshots**
    *   A Celery-backed worker polls news for subscribed topics, appending new evidence to the global database incrementally without full re-runs.

## 🚧 Ongoing Improvements
- **NLI Contradiction Engine:** Implementing DeBERTa-v3 to verify if new evidence *supports* or *contradicts* a canonical claim, rather than just clustering it.
- **RAG-based Chatting:** Integrating retrieval-augmented generation to allow users to "chat" directly with the extracted claims for a given event.
- **Deep Validation:** Integrating advanced prompt pipelines to penalize hallucinations in the claim extraction phase.

## 💻 Local Development Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database (Local or managed e.g., Supabase)
- Redis server (For Celery tasks)
- Valid API Keys for NewsAPI and HuggingFace

### 2. Installation
```bash
git clone https://github.com/kankaniakshat185/biasscope-hf-backend.git
cd biasscope-hf-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL="postgresql://user:password@localhost:5432/biasscope"
NEWS_API_KEY="your-news-api-key"
HF_TOKEN="your-huggingface-token"
REDIS_URL="redis://localhost:6379"
```

### 4. Database Push & Prisma Gen
```bash
prisma generate
prisma db push
```

### 5. Running the Application
**Start the API:**
```bash
uvicorn app.main:app --reload --port 8000
```
**Start the Celery Worker (In a separate terminal):**
```bash
celery -A app.celery_app worker --loglevel=info
```
