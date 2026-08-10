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

A claim-centric news intelligence engine. Instead of scoring an article's overall tone, BiasScope extracts individual factual claims from ingested articles, deduplicates and clusters them across publishers, and mathematically scores the resulting dataset's quality, polarization, and diversity — so a "balanced coverage" or "consensus" label means something checkable, not a vibe.

> **Looking for the Frontend UI?**
> [**BiasScope App Frontend**](https://github.com/kankaniakshat185/biasscope-app-frontend) — the Next.js dashboard that consumes this API.

[Live API Documentation](https://huggingface.co/spaces/kankaniakshat185/biasscope) • [Frontend Dashboard](https://biasscope-app.vercel.app/) • [Full Methodology & Trust Report](https://biasscope-app.vercel.app/#methodology)

## Features

- **Claim-centric extraction** — articles are broken into atomic, verifiable claims via a single cached LLM call, then run through a heuristic quality gate (word-boundary regex rejects questions, opinion language, and journalist commentary before anything is scored).
- **Two-tier deduplication** — near-identical claims within one article are merged (cosine similarity > 0.92); claims about the same fact across different articles in the *same search topic* are merged into one claim with multiple evidence rows (cosine similarity > 0.88, scoped by `search.query` so a claim can never silently absorb evidence from an unrelated topic).
- **Claim clustering & event detection** — HDBSCAN over a cosine distance matrix groups claims into candidate events; a mean pairwise-cohesion gate (≥ 0.72) rejects clusters that only share a topic, not a specific incident, and an eligibility gate requires ≥ 2 sources, ≥ 2 claims, and ≥ 2 evidence rows before something is surfaced as an "Event."
- **Cross-source consensus & contradiction scoring** — a DeBERTa-v3 NLI cross-encoder checks claim pairs within a cluster for genuine logical contradiction (not just disagreement in tone), which directly penalizes that cluster's consensus score.
- **Contrastive echo chambers** — separate, cached LLM-generated summaries of how left-leaning and right-leaning coverage frame the same topic.
- **Mathematically defined metrics, not heuristics dressed up as scores** — Data Quality Score (weighted completeness/diversity/richness), Polarization Score (Jensen-Shannon Divergence between ideological sentiment distributions, returned as `null` rather than a misleading 0% when one side has no coverage), and a Diversity Quality Label gated on source count, geographic spread, and ideological concentration.
- **Weekly topic snapshots** — a Celery/Redis background job delta-ingests new coverage for subscribed topics and tracks bias-distribution drift over time.
- **Every formula is documented against the real code** — see the [Methodology & Trust Report](https://biasscope-app.vercel.app/#methodology) on the frontend for the literal implementation behind every score, not a simplified explainer.

## Architecture

- **Compute:** a single FastAPI application (Uvicorn) deployed as a Docker container on Hugging Face Spaces.
- **Data:** PostgreSQL with the `pgvector` extension (hosted on Neon), storing articles, insights, claims, evidence, and 384-dimensional claim embeddings for cosine similarity search — managed through Prisma's Python client.
- **LLM access:** every call — extraction, narrative generation, canonicalization, chat — is routed through one client (`app/services/llm_client.py`) that SHA-256 hashes the prompt and checks a Postgres-backed cache before calling the Hugging Face Inference Router (`Qwen/Qwen2.5-7B-Instruct`), so identical prompts are never billed or re-run twice, and every call shows up in `/debug/llm-usage`.
- **Auth:** the frontend's Better Auth session cookie is the source of truth. This backend never issues its own tokens — `app/deps/auth.py` looks the session token up directly in the `session` table Better Auth already writes to, in the same database.
- **Cross-origin access:** the frontend does not call this API directly from the browser. Hugging Face Spaces' front-door proxy strips `Access-Control-Allow-Credentials` on cross-origin preflight requests (a platform-level policy, not something fixable in this app's CORS config), so the frontend routes every request through its own same-origin Next.js API relay, which forwards to this backend server-to-server instead.
- **Background jobs:** Celery + Redis (Upstash) for the weekly snapshot task that powers longitudinal topic tracking under Subscriptions.

<details>
<summary><b>View Detailed Architecture Diagram</b></summary>

```mermaid
graph TD
    Browser[Browser] -->|same-origin, credentialed| Proxy["Next.js /api/proxy\n(server-to-server relay)"]
    Proxy -->|HTTPS + session cookie| API[FastAPI on Hugging Face Spaces]

    API --> Auth["app/deps/auth.py\nsession lookup"]
    API --> Pipeline["run_search_pipeline\n(app/services/pipeline.py)"]

    subgraph "Synchronous /search"
        Pipeline --> Ingest["ingest_articles\nNewsAPI + GDELT + newspaper3k"]
        Ingest --> Clean["clean_and_deduplicate\nURL + fuzzy-title dedup"]
        Clean --> NLP["analyze_articles\nsentiment / bias / NER"]
        NLP --> Validate["validate_articles\nDQS, JSD polarization, diversity"]
        Validate --> Narrative["generate_narrative +\ncontrastive summaries (LLM, cached)"]
        Narrative --> Persist[(Search / Article / Insight)]
    end

    Persist -.->|BackgroundTasks, non-blocking| Phase2["run_phase2_pipeline"]

    subgraph "Background Phase 2 — phase2Status: pending -> processing -> complete"
        Phase2 --> Extract["process_and_store_claims\nquality gate + dedup"]
        Extract --> Cluster["run_claim_clustering\nHDBSCAN, scoped to this topic"]
        Cluster --> Events["run_event_detection\ncohesion + eligibility gates, NLI contradiction check"]
        Events --> ClaimDB[(Claim / Evidence / ClaimCluster / Event)]
    end

    Extract -.-> LLMClient["cached_llm_call\nSHA-256 prompt cache"]
    Narrative -.-> LLMClient
    LLMClient -->|cache miss| HFRouter["HF Inference Router\nQwen2.5-7B-Instruct"]
    LLMClient --> LLMCacheDB[(LLMCache / LLMUsage)]

    ClaimDB -->|GET /results/:id/intelligence| API
    Persist -->|GET /results/:id| API

    Snapshot["Celery Beat: run_weekly_snapshots\n(app/tasks/snapshot_task.py)"] -.->|weekly, per subscription| Ingest
    Snapshot --> SnapshotDB[(TopicSnapshot)]
    Redis[(Upstash Redis)] --- Snapshot
```
</details>

## Local Setup & Installation

Python 3.10+ and a PostgreSQL database with the `pgvector` extension (e.g. [Neon](https://neon.tech)).

### 1. Clone and install
```bash
git clone https://github.com/kankaniakshat185/biasscope-hf-backend.git
cd biasscope-hf-backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the repo root:
```env
# Database (Prisma reads this directly — must support pgvector)
DATABASE_URL="postgresql://user:password@host:port/dbname?sslmode=require"

# News ingestion
NEWS_API_KEY="your_newsapi_key"

# LLM inference — used for extraction, narrative generation, and chat
HF_TOKEN="your_huggingface_token"

# Background jobs (only needed if you're running the Celery worker/beat — see Testing below)
REDIS_URL="rediss://default:password@host:port"

# CORS allow-list — comma-separated, defaults to localhost:3000 + the production frontend
ALLOWED_ORIGINS="http://localhost:3000,https://biasscope-app.vercel.app"

# Optional: exposes /debug/* (cluster/event inspection, cache stats, rerun tools).
# Requires a valid session on top of this flag — never set this on a public deployment.
ENABLE_DEBUG_ROUTES="1"
```

### 3. Generate the Prisma client and sync the schema
```bash
prisma generate
prisma db push
```

### 4. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Swagger docs at `http://127.0.0.1:8000/docs`.

### 5. (Optional) Run the Celery worker
`celery` and `redis` are declared dependencies, but no worker process is deployed in production today — the weekly snapshot job's code path is fully tested (see Testing below) but doesn't currently run anywhere unattended. To exercise it locally:
```bash
celery -A app.celery_app worker --pool=solo --loglevel=info
celery -A app.celery_app beat --loglevel=info   # if you want the weekly schedule to actually fire
```

## Project Structure

```text
biasscope-hf-backend/
├── app/
│   ├── main.py               # App wiring only — CORS, router registration, startup/shutdown
│   ├── db.py                 # Shared Prisma client singleton
│   ├── celery_app.py         # Celery app + Redis connection config
│   ├── deps/
│   │   └── auth.py           # Session-cookie auth (reads Better Auth's session table directly)
│   ├── routers/               # search, results, subscriptions, history, chat, debug
│   ├── services/
│   │   ├── pipeline.py        # /search orchestration + the Phase 2 background pipeline
│   │   ├── intelligence.py    # Read-side claim/cluster/event graph queries
│   │   ├── ingestion.py       # NewsAPI + GDELT fetching, newspaper3k scraping
│   │   ├── cleaning.py        # URL + fuzzy-title deduplication
│   │   ├── nlp.py             # Sentiment, bias classification, NER, entity-sentiment graph
│   │   ├── extraction.py      # Claim extraction, quality gate, embedding-based dedup
│   │   ├── clustering.py      # HDBSCAN clustering, event detection, consensus/importance scoring
│   │   ├── validation.py      # DQS, JSD polarization, diversity label
│   │   └── llm_client.py      # Cached, usage-tracked LLM client — every call goes through here
│   ├── tasks/
│   │   └── snapshot_task.py   # Weekly Celery Beat job for topic-subscription drift tracking
│   ├── utils/                 # One-off maintenance scripts (index creation, orphan cleanup, resets)
│   └── prisma_client/         # Generated Prisma Python client (not hand-written)
├── prisma/
│   └── schema.prisma          # Relational schema + pgvector embedding column
├── tests/                     # 220+ tests — see Testing below
├── learning_notes/            # Development notes and ADRs
├── AUDIT_TASKS.md             # Running log of engineering audits and remediation work on this repo
├── pyproject.toml             # ruff + mypy + pytest configuration
├── requirements.txt
└── requirements-dev.txt       # ruff, mypy, pytest, pytest-cov, pytest-asyncio
```

## Code Quality

```bash
pip install -r requirements-dev.txt
ruff check .      # lint
mypy app/         # type check (generated Prisma client excluded, its types still used elsewhere)
```
Both are configured in `pyproject.toml` and kept clean as part of this repo's normal workflow — see `AUDIT_TASKS.md` for the history of what each pass caught.

## Testing

```bash
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm   # once, for the real-model tests below

pytest                    # full suite
pytest -m "not model"     # skip tests that load real ML models (faster, no download needed)
pytest --cov=app --cov-report=term-missing
```

220+ tests, covering every hand-written file in `app/` (the generated `app/prisma_client/` is the only exclusion). Real, not mocked, ML models are used wherever the thing under test genuinely is the model's behavior — sentence-transformers for embedding similarity thresholds, the `cross-encoder/nli-deberta-v3-small` cross-encoder for contradiction/grounding checks. Nothing runs against a real Postgres/pgvector database (there isn't one available in this environment); DB-touching code — including the weekly snapshot task and every script in `app/utils/` — is tested against a hand-built fake Prisma client (`tests/fakes.py`) with real business logic exercised on top of it.

Several real, previously-shipped bugs were caught this way rather than by manual testing: a `POST /subscriptions` request-shape regression, a route-registration crash from a mistyped type hint, an evidence-count double-counting bug, a snake_case/camelCase key mismatch that made every weekly snapshot's bias/polarization numbers permanently wrong, and a `SELECT DISTINCT` + `ORDER BY` combination that's syntactically valid Python but an actual Postgres error. See `AUDIT_TASKS.md` for the full list.

## Privacy & License

See [`PRIVACY.md`](./PRIVACY.md) for what data this service processes and why. Licensed under the [MIT License](./LICENSE).
