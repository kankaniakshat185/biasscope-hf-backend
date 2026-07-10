---
title: Biasscope
emoji: 🦀
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: false
---

# BiasScope Backend Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Prisma](https://img.shields.io/badge/Prisma-3982CE?style=for-the-badge&logo=Prisma&logoColor=white)](https://prisma.io/)

A high-performance, claim-centric natural language processing engine that powers the BiasScope Intelligence Dashboard. This backend drives macro-level media bias analysis by scraping, cleaning, clustering, and evaluating thousands of news articles in real-time.

> **Looking for the Frontend UI?**
> The frontend application that pairs with this backend can be found here: 
> [**BiasScope App Frontend**](https://github.com/kankaniakshat185/biasscope-app-frontend)

[Live API Documentation](https://huggingface.co/spaces/kankaniakshat185/biasscope) • [Frontend Dashboard](https://biasscope-app-frontend.vercel.app/)

## 🚀 Features

- **Claim-Centric Ingestion Pipeline** — Distills raw articles into discrete, factual claims rather than relying on noisy article-level sentiment.
- **Semantic Claim Clustering** — Utilizes `sentence-transformers/all-MiniLM-L6-v2` and `pgvector` to map semantically equivalent claims into unified entities across multiple publications.
- **Cross-Ideological Consensus Engine** — Programmatically evaluates the publisher diversity for individual claims to detect and flag corroborated narratives vs. isolated partisan talking points.
- **Contrastive Echo Chambers** — Isolates political ecosystems to generate distinct, sophisticated LLM-driven analyses of how identical events are framed by different sides of the political spectrum.
- **Automated Topic Snapshots** — Redis-backed Celery workers incrementally append new evidence to the global database without redundant reprocessing, allowing instant demo generation.
- **Robust Metrics Validation** — Mathematically derives Polarization Scores using Jensen-Shannon Divergence and calculates Data Quality Scores based on completeness, source diversity, and content richness.

## 🏗️ Architecture

BiasScope operates entirely in the cloud, utilizing a decoupled, edge-ready architecture:
- **Compute Layer:** Containerized FastAPI instances deployed on HuggingFace Spaces.
- **Data Persistence:** Managed PostgreSQL (NeonDB) instances handling thousands of vector embeddings and relational entities simultaneously.
- **Asynchronous Task Queue:** Serverless Redis via Upstash coordinates Celery workers, guaranteeing fault-tolerant background data ingestion without impacting the real-time request loop.

<details>
<summary><b>View Detailed Architecture Diagram</b></summary>

```mermaid
graph TD
    subgraph Client
        UI[BiasScope Frontend]
    end

    subgraph API Layer
        API[FastAPI Server]
        Redis[Upstash Redis Queue]
    end

    subgraph Worker Layer
        Worker[Celery Worker]
        NewsAPI[NewsAPI Aggregator]
        Clean[Data Cleaning & Deduplication]
        LLM[Llama-3 Claim Extraction Engine]
        Embed[SentenceTransformers Embedding]
    end

    subgraph Data Layer
        DB[(PostgreSQL w/ pgvector)]
    end

    UI -->|POST /search| API
    API -->|Enqueue Task| Redis
    Redis -->|Consume Task| Worker
    
    Worker -->|Fetch Articles| NewsAPI
    Worker --> Clean
    Clean --> LLM
    LLM --> Embed
    Embed -->|Upsert Claims| DB
    
    DB -->|Read Claims| Worker
    Worker -->|Cosine Similarity Clustering| DB
    
    Worker -->|Generate Intelligence Report| DB
    API -->|Fetch Report| DB
    API -->|Return JSON| UI
```
</details>

## 💻 Local Setup & Installation

To run the BiasScope backend locally, you will need Python 3.10+ and a PostgreSQL database (e.g., NeonDB) supporting `pgvector`.

### 1. Clone the repository
```bash
git clone https://github.com/kankaniakshat185/biasscope-hf-backend.git
cd biasscope-hf-backend
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add the following keys:
```env
# Database
DATABASE_URL="postgresql://user:password@host:port/dbname?sslmode=require"

# Message Broker
CELERY_BROKER_URL="rediss://default:password@host:port"
CELERY_RESULT_BACKEND="rediss://default:password@host:port"

# API Keys
NEWS_API_KEY="your_newsapi_key"
GROQ_API_KEY="your_groq_api_key"
HF_TOKEN="your_huggingface_token" # Optional, for higher rate limits
```

### 5. Generate Prisma Client & Push Schema
Initialize the database tables and generate the Python Prisma client:
```bash
prisma generate
prisma db push
```

### 6. Run the Application
You need to run both the FastAPI server and the Celery worker concurrently.

**Terminal 1 (FastAPI Server):**
```bash
set -a && source .env && set +a
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 (Celery Worker):**
```bash
set -a && source .env && set +a
celery -A app.celery_app worker --pool=solo --loglevel=info
```

The API will now be available at `http://127.0.0.1:8000`. You can view the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## 📁 Project Structure

```text
biasscope-hf-backend/
├── app/
│   ├── main.py                 # FastAPI entry point & API routes
│   ├── celery_app.py           # Celery background task configuration
│   ├── prisma_client/          # Auto-generated Prisma ORM client
│   └── services/
│       ├── ingestion.py        # External data fetchers (NewsAPI, GDELT)
│       ├── cleaning.py         # Text deduplication and sanitization
│       ├── extraction.py       # Llama-3 based atomic claim extraction
│       ├── nlp.py              # Sentiment analysis & Echo Chamber logic
│       ├── clustering.py       # Vector embeddings & Cosine similarity merging
│       └── validation.py       # Mathematical polarization and DQS formulas
├── prisma/
│   └── schema.prisma           # Relational schema & pgvector definitions
├── learning_notes/             # Extensive documentation & ADRs for personal reference
├── tests/                      # Evaluation framework and unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Performance & Optimization

- **Claim Extraction Throughput:** By decoupling extraction from blocking HTTP requests, the pipeline achieves an average extraction time of 2.3s per batch.
- **Vector Operations:** Embedding generation takes ~45ms per claim, and consensus calculations resolve in ~12ms per event utilizing optimized `pgvector` indexing.
- **Actor Model Optimization:** The adoption of a distributed actor model for the Llama 3 endpoint reduced LLM timeout rates from 14% to 0%, increasing throughput by 3x during peak news cycles.

## 🧪 Evaluation Framework

BiasScope utilizes a rigorous testing pipeline to validate the NLP engine.

**Test 1: Hallucination Penalty**
Runs extracted claims through a cross-encoder to verify that the LLM output is 100% grounded in the original source sentence.
```bash
pytest tests/nlp/test_grounding.py
```

**Test 2: Clustering Thresholds**
Validates the cosine similarity threshold (0.85) to ensure distinct claims are not improperly merged into the same canonical claim.
```bash
pytest tests/clustering/test_similarity.py
```

## 📜 License

MIT License
