# BiasScope — Intelligence Engine Backend

A multi-source news intelligence platform that extracts verifiable factual claims from news articles, clusters them across publishers, and surfaces cross-source consensus events.

## Architecture

```
Search Query
    ↓
Article Ingestion (NewsAPI + Scraping)
    ↓
NLP Analysis (Sentiment, Bias, NER)
    ↓
Claim Extraction (LLM — cached)
    ↓
Quality Gates (type, quality, relevance)
    ↓
Vector Embeddings (all-MiniLM-L6-v2)
    ↓
Claim Clustering (HDBSCAN)
    ↓
Cohesion Validation (pairwise cosine sim)
    ↓
Cross-Source Validation (sources ≥ 2)
    ↓
Event Detection & Ranking
    ↓
Intelligence Report
```

## Why Source Diversity Matters

A claim reported by a single source could be a press release, a rumor, or an error.
When **multiple independent publishers** converge on the same factual claim, it becomes a verified intelligence signal.

BiasScope enforces a strict **multi-source convergence gate**:
- A cluster must have **≥ 2 unique sources**, **≥ 2 claims**, and **≥ 2 evidence records** to become an Event.
- Single-source echo chambers are rejected regardless of claim volume.

## Why Evidence Attribution Exists

Every stored claim links back to the **specific article, sentence, source domain, and URL** it was extracted from.
This enables:
- **Provenance tracking** — trace any claim to its origin.
- **Cross-source scoring** — measure how many independent publishers confirm the same fact.
- **Bias-aware analysis** — compare how left, center, and right sources frame the same event.

## Pipeline Details

### Extraction Gates
| Gate | Method | Purpose |
|------|--------|---------|
| 0 | Cosine deduplication (> 0.92) | Remove within-article duplicates |
| 1 | Claim type filter | Keep only EVENT and NUMERIC types |
| 2 | Quality score (heuristic) | Reject opinions, biography, commentary |
| 3 | Embedding relevance (> 0.40) | Reject claims unrelated to query |

### Clustering
- **HDBSCAN** with cosine distance matrix and leaf selection for finest granularity.
- **Cohesion validation**: Clusters with mean pairwise similarity below 0.65 are rejected (prevents topic-level merging).

### Event Detection
- Cross-source consensus scoring: `unique_sources / claim_count`.
- Weighted importance ranking incorporating source count, publisher diversity, evidence volume, and consensus quality.

## LLM Usage

| Call | Stage | When | Cached |
|------|-------|------|--------|
| Claim Extraction | `extraction` | 1× per article | ✅ SHA-256 |
| Canonical Claim | `canonicalization` | 1× per cluster | ✅ SHA-256 |
| Event Summary | `event_summary` | 1× per event | ✅ SHA-256 |

All calls use `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API.
Prompt-level SHA-256 caching ensures zero duplicate API calls.

## Utility Commands

```bash
# Reset claim graph (claims, clusters, events, evidence, cache)
python -m app.utils.reset_claim_graph

# Reset ALL data (includes articles, insights, searches)
python -m app.utils.reset_claim_graph --all
```

## Tech Stack

- **Backend**: FastAPI + Python
- **Database**: PostgreSQL + pgvector (Neon)
- **ORM**: Prisma (Python client)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Clustering**: HDBSCAN (scikit-learn)
- **NLP**: Spacy (en_core_web_trf), DistilBERT sentiment, PoliticalBiasBERT
- **LLM**: Qwen 2.5 7B Instruct (HuggingFace Inference)
- **Deployment**: HuggingFace Spaces (Docker)
