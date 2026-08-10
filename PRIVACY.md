# Privacy Policy

**Last updated:** August 2026

This document covers what the BiasScope **backend API** processes. If you're an end user of the product, the [frontend's privacy policy](https://github.com/kankaniakshat185/biasscope-app-frontend/blob/main/PRIVACY.md) is the authoritative, user-facing version — this file exists for anyone deploying, auditing, or building against this API directly.

## 1. What this service processes

This is a stateless-per-request API with a Postgres database behind it. Per request, it handles:

- **Session cookies** — every authenticated route reads the Better Auth session cookie already set by the frontend and looks it up directly in the shared `session` table (`app/deps/auth.py`). This service never issues its own tokens and has no separate login flow.
- **Search queries** — the topic text you search for, sent to NewsAPI and GDELT to source articles, and to the Hugging Face Inference Router for LLM analysis.
- **Article content** — full text scraped from public news URLs (via `newspaper3k`), analyzed and stored (sentiment, bias classification, entities, extracted claims) tied to the search that produced it.
- **LLM prompts and responses** — every LLM call is SHA-256 hashed and cached in Postgres (`LLMCache`) with per-call token/cost metadata (`LLMUsage`), so repeat prompts across users are never re-sent to the model provider.

## 2. Where data goes

| Destination | What it receives | Purpose |
|---|---|---|
| **Hugging Face Inference Router** | Article text, extracted claims, generated prompts | Runs `Qwen/Qwen2.5-7B-Instruct` for claim extraction, narrative generation, and contrastive summaries. |
| **NewsAPI / GDELT** | The search query (topic keywords only) | Article sourcing. |
| **Neon (PostgreSQL + pgvector)** | Everything persisted — articles, claims, evidence, embeddings, sessions, cache | Primary datastore. |
| **Upstash (Redis)** | Job metadata for the weekly snapshot task | Backs the Celery queue for topic-subscription drift tracking, when that worker is running. |
| **Hugging Face Spaces** | The running container itself | Hosting; standard infrastructure request logs apply per [Hugging Face's privacy policy](https://huggingface.co/privacy). |

No analytics, telemetry, or error-monitoring SDK is integrated into this service — nothing here reports usage data anywhere beyond the `LLMUsage` table this service's own operator can query via `/debug/llm-usage`.

## 3. Debug routes

`ENABLE_DEBUG_ROUTES=1` exposes a set of `/debug/*` endpoints (cluster/event inspection, cache stats, manual pipeline reruns) intended for the operator only, gated behind the same session-cookie auth as everything else. This flag should never be set to `1` on a deployment anyone other than the operator can reach.

## 4. Data retention & deletion

Data lives in Postgres until deleted. There's no automatic expiry. To request deletion of data associated with a specific account, contact the operator (see below) — this backend has no self-service deletion endpoint today.

## 5. Contact

**kankaniakshat185@gmail.com**

---

See also: [MIT License](./LICENSE) · [BiasScope App Frontend](https://github.com/kankaniakshat185/biasscope-app-frontend)
