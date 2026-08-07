# BiasScope Remediation Tasklist

Condensed from the [full engineering audit](https://claude.ai/code/artifact/1a32c5de-9e38-4d5d-a473-73c56562a652) (26 findings). Grouped by theme, ordered by severity within each group. Repo tag shows where the fix lives: `[BE]` = biasscope-hf-backend, `[FE]` = biasscope-app-frontend.

Check items off as they land. Security & Auth is the current focus — start there.

## 🔴 Security & Auth — DO FIRST

- [x] **S1** `[BE]` Add auth to every route; gate/remove `/debug/*` endpoints in prod — extracted to `app/routers/debug.py`, gated behind `ENABLE_DEBUG_ROUTES=1` + login (no role-based admin check yet — see follow-up note below)
- [x] **S2** `[BE]`/`[FE]` Fix IDOR — `app/deps/auth.py` resolves the user from the Better Auth session cookie; `/search`, `/subscriptions`, `/history` no longer accept a client-supplied `userId`; added an ownership check to `DELETE /history/{id}` (was previously missing entirely)
- [x] **S1b** `[BE]` CORS now reads `ALLOWED_ORIGINS` (comma-separated env var, defaults to localhost + the Vercel URL) instead of `"*"`
- [x] **S3** `[BE]` Vector dedup query in `extraction.py` now uses `$1`/`$2` bound parameters instead of f-string interpolation
- [x] **S4** `[BE]` `prisma db push` moved out of the FastAPI startup handler into the Docker `CMD` (`&&`-chained so a failed sync stops the container instead of failing silently)
- [x] **S5** `[FE]` `BETTER_AUTH_SECRET` now throws at boot if unset; also fixed the adjacent `PrismaClient` hot-reload singleton bug (A6) and added cross-origin cookie attributes for prod (`advanced.defaultCookieAttributes`) while in the file

**Follow-ups from this pass (not yet done):**
- No role-based admin gating exists (`User` has no `role` column) — `ENABLE_DEBUG_ROUTES` currently means "any logged-in user," not "admins only." Fine for a locked-down solo/staging deploy, not for a multi-user prod environment.
- **Verify locally before deploying:** the auth dependency assumes Better Auth's default cookie name (`better-auth.session_token` / `__Secure-` variant) and the `advanced.defaultCookieAttributes` config key — confirm both by inspecting a real login's `Set-Cookie` header against your installed better-auth version.
- `/results/{id}`, `/results/{id}/intelligence`, `/demo/{topic}`, `/chat-with-article`, `/chat-with-summary` are still unauthenticated (read-only / non-destructive; a guessable `search_id` can still be read by anyone). Worth a login requirement later but was out of scope for this pass.

## 🟠 Architecture & Layering — DONE

- [x] **A1** `[BE]` `main.py` shrunk from 828 lines to ~70 (app wiring only). New structure: `app/routers/{search,results,subscriptions,history,chat,debug}.py` for routes, `app/services/{pipeline,intelligence}.py` for the business logic they call.
- [x] **A2** `[BE]` `get_results`/`get_search_intelligence` now live in `app/services/intelligence.py`; `snapshot_task.py`'s dead `from app.main import get_search_intelligence` is gone; `create_demo_snapshot.py` imports from the service module + `app.db` instead of `app.main`.
- [x] **A3** `[BE]`/`[FE]` `get_results()` now attaches `reliabilityScore`/`reliabilityTier` to every article from the backend's `SOURCE_RELIABILITY` registry; the frontend's independent (and disagreeing) hardcoded domain arrays in `dashboard/[id]/page.tsx` are deleted — it renders whatever tier the API says now.
- [x] **A4** `[BE]` `cached_llm_call` takes a `model` param; `generate_narrative`/`generate_contrastive_summaries` (nlp.py) and the two chat endpoints (now unified into one `_chat_with_context` helper in `app/routers/chat.py`, fixing the Q3 duplication too) all route through it. Every LLM call in the app now shows up in `/debug/llm-usage` and `/debug/cache-stats`.
- [x] **A5** `[FE]` Frontend's `prisma/schema.prisma` slimmed to Better Auth tables only (`User`/`Session`/`Account`/`Verification`) — `Search`/`Article`/`Insight` deleted from it entirely, with a comment explaining the backend owns those tables and this schema must only ever run `prisma generate`, never `migrate`/`db push`. Verified no frontend code referenced those models before removing them, and re-validated the schema afterward (`prisma validate` passes). **The underlying database tables were never touched** — this was a client-schema-only change.
- [x] **A6** `[FE]` Done alongside S5 — `globalForPrisma` singleton guard added in `lib/auth.ts`.

**Note:** `app/services/pipeline.py`'s `run_search_pipeline` still raises `HTTPException` directly (matching the original behavior) — a stricter clean-architecture split would have services raise domain exceptions and let routers translate them to HTTP, but that's a bigger change than this pass scoped for.

## 🟠 Data Integrity & Performance

- [x] **P1** `[BE]` `run_claim_clustering(prisma, query)` now scopes to claims whose evidence traces back to the same `search.query` (case-insensitive) — bounds cost to one topic's claim volume instead of the whole table, while still preserving cross-search consensus *within* a topic. `/debug/rerun-clustering` intentionally keeps the old global behavior (`query=None`) since it wipes and rebuilds every cluster anyway.
- [x] **D4** `[BE]`/`[FE]` `/history` now takes `limit`/`offset` (default 50, cap 200) and returns `{total, limit, offset, searches}`; frontend updated to match. **Deliberately not done** for `/results/{id}` and `/results/{id}/intelligence` — both are already naturally bounded by the ~50-article ingestion cap per search, so they don't have `/history`'s actual unbounded-growth problem; reshaping their response would touch a lot of frontend surface for little real protection. Revisit if that assumption stops holding (e.g. if repeated re-extraction on one search_id starts meaningfully growing its claim count).
- [x] **D1** `[BE]` Added `article Article @relation(..., onDelete: Cascade)` to `Evidence` in `schema.prisma`. **Needs a manual step before this is live** — see below.
- [x] **D2** `[BE]` Wrote `app/utils/create_vector_index.py` (HNSW, cosine ops, matching the `<=>` operator already used everywhere). **Needs a manual step before this is live** — see below.
- [x] **D3** `[BE]` Dropped `ConsensusFact`/`ContradictionPair`/`SnapshotEvent`/`SnapshotClaim` from `schema.prisma` per your call — removed the now-dead `.contradictionpair`/`.consensusfact` calls in `reset_claim_graph.py` and updated the stale comment in `snapshot_task.py` that referenced the deleted `SnapshotEvent` table.

**Manual steps required (I have no access to your live database from here):**
1. Run `python -m app.utils.cleanup_orphaned_evidence` (dry run first, then `--delete`) — **must** show 0 orphans before the next step, or it will fail.
2. Run `python -m app.utils.create_vector_index` once.
3. Run `prisma db push` (or your normal deploy) to apply the `Evidence.articleId` FK and the dropped Phase 3 models to the real schema. Since step 1 guarantees no orphans, the `ADD CONSTRAINT` should apply cleanly — but do this on a moment you can watch the logs, in case anything about your specific data surprises it.

## 🟡 Dead Code

- [ ] **X1** `[BE]` Delete `scrape_single_url`/`extract_text_from_images` + unused deps (`trafilatura`, `pytesseract`, `Pillow`, `beautifulsoup4`) + `tesseract-ocr` apt package — `app/services/ingestion.py:191-315`, `Dockerfile:8`, `requirements.txt`
- [ ] **X2** `[FE]` Delete orphaned `lib/auth-client.ts` (root) — unused duplicate of `src/lib/auth-client.ts`
- [ ] **X3** `[FE]` Delete `README 2.md`
- [ ] **X4** `[BE]` Rename `test_celery.py` → `scripts/trigger_weekly_snapshot.py` (it's not a test, will break pytest collection)

## 🟡 Code Quality

- [ ] **Q1** `[BE]` Word-boundary the opinion/biographical/commentary signal matching (currently bare substring) — `app/services/extraction.py:149-192`
- [ ] **Q2** `[BE]` Standardize on `logging` everywhere (currently split with `print()`); add a minimal error-reporting sink
- [ ] **Q3** `[BE]` Dedupe `/chat-with-article` and `/chat-with-summary` into one helper; drop redundant local `import os` — `app/main.py:430-467,469-503`
- [ ] **Q4** `[BE]` Drop defensive `getattr()` calls once S4 stops schema drift — `app/main.py:321-322,338-339,362-364`

## 🟡 Frontend

- [ ] **F1** `[FE]` Centralize `NEXT_PUBLIC_BACKEND_URL` fallback into one `lib/api.ts` client instead of ~10 inline copies
- [ ] **F2** `[FE]` Remove `console.log("Bias data:", data)` — `src/components/Charts.tsx:62`
- [ ] **F3** `[FE]` Fix README's `vault/` route reference (actual route is `history/`) and missing `docs/dashboard-preview.png`
- [ ] **F4** `[FE]` Replace unbounded 10s polling with a real pipeline-status flag from the backend — `IntelligenceLayer.tsx:37-40`

## 🟡 Testing & Docs

- [ ] **T1** `[BE]`/`[FE]` Build the test suite the backend README already claims exists (`tests/nlp/test_grounding.py`, `tests/clustering/test_similarity.py`) — highest-value gap in the whole audit
- [ ] **T2** `[BE]` Remove the dead `learning_notes/` rule from `.gitignore` (folder is tracked anyway)

## 🟢 DevOps

- [ ] **V1** `[BE]` Remove or document which deploy config is canonical — `vercel.json` vs. `Dockerfile`/HF Spaces

---
**Suggested order:** Security & Auth → Data Integrity → Architecture → everything else. See the [full audit](https://claude.ai/code/artifact/1a32c5de-9e38-4d5d-a473-73c56562a652) for rationale, code excerpts, and effort estimates per item.
