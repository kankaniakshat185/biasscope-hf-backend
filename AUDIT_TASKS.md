# BiasScope Remediation Tasklist

Condensed from the [full engineering audit](https://claude.ai/code/artifact/1a32c5de-9e38-4d5d-a473-73c56562a652) (26 findings). Grouped by theme, ordered by severity within each group. Repo tag shows where the fix lives: `[BE]` = biasscope-hf-backend, `[FE]` = biasscope-app-frontend.

Check items off as they land. Security & Auth is the current focus — start there.

## 🐛 Bugs found while writing the T1 test suite

Not audit findings — these were introduced or exposed by earlier fixes in this same remediation pass, and none were caught by `ruff`/`mypy`. Listed here because they were live in the working tree until the test suite caught them.

1. **`POST /subscriptions` was broken** — removing `userId` as a sibling `Body(...)` field during the S2 fix left `topic` as the only `Body` param, which silently changed FastAPI's expected request shape from `{"topic": "..."}` to a bare string. The frontend still sends the wrapped shape. Fixed with `Body(..., embed=True)`. **If you deployed the S2 change before this fix, subscribing to a topic has been failing with a 422 since then.**
2. **The app couldn't start** — a mypy cleanup pass changed `background_tasks: BackgroundTasks = None` to `BackgroundTasks | None = None` in `search.py` to satisfy mypy's implicit-optional check. FastAPI special-cases the *exact* bare `BackgroundTasks` type to auto-inject an instance; wrapped in `Optional`, route registration crashes at import time. Reverted to the bare form with a `# type: ignore` and a comment explaining why it must stay that way.
3. **Cluster `evidenceCount` over-counted** (pre-existing, not introduced this session) — `get_search_intelligence()` computed a cluster's `evidenceCount` *before* deduplicating evidence by sentence, while the equivalent code for events computed it *after* — an inconsistency that made a cluster report a higher evidence count than the number of evidence items actually returned. Fixed to match the (correct) events behavior.

## 🐛 Bugs found from live production logs (2026-08-08)

Found by reading actual HF Space runtime logs during a real `/search` run, not by the test suite. Regression tests added for all three (`tests/routers/test_search_router.py`, `tests/nlp/test_narrative_fallback.py`, `tests/test_llm_client.py`).

1. **Every `/search` request 422'd whenever the frontend's category dropdown was left unset.** `page.tsx` sends `category: category || undefined`, which `JSON.stringify` drops from the body entirely — but the router declared `category: str = Body(...)` (required). `ingestion.py` already treats `""`/`"all"` as "no filter," so the fix was just changing the router default to `Body("")` to match. **Fixed in `app/routers/search.py`.**
2. **`NARRATIVE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"` is no longer served by any provider enabled on the HF Inference Router** — every narrative/chat LLM call has been failing with a 400 (`model_not_supported`) and silently falling back. Pinned `NARRATIVE_MODEL_ID = MODEL_ID` (the Qwen model the extraction stage already calls successfully) instead of a second, independently-rotting model id. **Fixed in `app/services/llm_client.py`.** Narrative/chat quality should be re-eyeballed against a real search now that it's actually reaching the LLM again.
3. **The narrative fallback crashed with an unhandled `TypeError` and took the whole `/search` request down with it**, whenever it ran (i.e. whenever bug #2 above was live, or `HF_TOKEN` was unset): `extract_keywords()` returns `[{"word": ..., "count": ...}, ...]` dicts, but `_generate_fallback_narrative` did `", ".join(keywords[:3])` as if they were plain strings. Silent in the test suite because the synthetic short titles used there never produced any real keywords, so the buggy line was never reached. **Fixed in `app/services/nlp.py`.**

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

## 🟡 Dead Code — DONE

- [x] **X1** `[BE]` Deleted `scrape_single_url`/`extract_text_from_images` (~120 lines) from `ingestion.py`; removed `trafilatura`, `pytesseract`, `Pillow`, `beautifulsoup4` from `requirements.txt`; removed `tesseract-ocr` from the `Dockerfile` apt install. Re-verified nothing else in the codebase referenced any of it before deleting.
- [x] **X2** `[FE]` Deleted the orphaned root `lib/auth-client.ts` — confirmed every real import resolves to `src/lib/auth-client.ts` first. `lib/` now contains only `auth.ts` (server), which was the actual intent all along.
- [x] **X3** `[FE]` Deleted `README 2.md`.
- [x] **X4** `[BE]` Moved `test_celery.py` → `scripts/trigger_weekly_snapshot.py`. Checked for other references to the old filename first (none, besides this file).

## 🟡 Code Quality

- [x] **Q1** `[BE]` `extraction.py`'s opinion/biographical/commentary signal lists now match via compiled `\b...\b` word-boundary regex instead of bare substring `in` checks — verified against real false-positive cases (`undamaged`, `uncontroversial`, `validating` no longer wrongly match; `grandchildren` no longer matches `children`).
- [x] **Q2** `[BE]` Converted every production `print()` (ingestion.py, nlp.py, routers/debug.py) to `logging` — kept `print()` only in `cohesion_analysis.py`/`scripts/trigger_weekly_snapshot.py`, which are CLI tools meant for direct terminal output, not logging. Also added `logging.basicConfig(...)` to `main.py`, without which `logger.info()` calls would have silently gone nowhere (Python's root logger defaults to WARNING with no handler; uvicorn configures its own loggers, not the app's). `logger.exception(...)` replaces the old `print(e); traceback.print_exc()` pattern in the debug router's background tasks, capturing the same info in one structured call. No third-party error-tracking service wired up (would need picking a vendor and credentials neither of us has right now) — that's the "minimal error-reporting sink" left undone, everything else in Q2 is complete.
- [x] **Q3** `[BE]` Already fixed during the A4 refactor — `/chat-with-article`/`/chat-with-summary` are now one shared `_chat_with_context` helper in `app/routers/chat.py`.
- [x] **Q4** `[BE]` Removed the no-op `getattr(obj, 'field', default)` wrappers in `app/services/intelligence.py` — simplified to `obj.field or default`, which was already doing the real work.

## 🔧 Tooling — added at your request (not an original audit finding)

- [x] `ruff` (lint) + `mypy` (type checking) configured in `pyproject.toml`; pinned in `requirements-dev.txt` (`pip install -r requirements-dev.txt`).
- [x] Fixed a real, previously-invisible gap: `app/__init__.py` and `app/services/__init__.py` didn't exist (every other subpackage had one) — this was confusing mypy's module resolution and is a genuine consistency fix, not just a tooling workaround.
- [x] `ruff check .` — started at 133 pre-existing issues (116 auto-fixed mechanically: whitespace, import order, modern type-hint syntax; 12 fixed by hand: one-line-if statements, an ambiguous variable name `l`, a bare `except:`, an unused variable, an unused loop variable). **Zero issues now.**
- [x] `mypy app/` — started at 305 pre-existing errors across 22 files once the `__init__.py` gap was fixed (mostly Prisma's nullable list relations needing an `... or []` guard before iterating, plus missing var-annotations). **Zero issues now.** Generated code (`app/prisma_client/`) is excluded from being *checked* but still used for accurate types elsewhere (an `ignore_errors` override, not a blanket `exclude`, which would have also hidden its types from everything that imports it).
- [x] **Found and fixed a real bug along the way**: `app/utils/create_demo_snapshot.py` was passing a pre-serialized JSON *string* into a field typed `Json`. The generated Prisma client only JSON-encodes values wrapped in `Json(...)` — a bare string falls through to default string serialization, so `DemoSnapshot.data` was being stored as a JSON-encoded string rather than a JSON object, meaning `/demo/{topic}` would have handed the frontend an escaped JSON string instead of a parsed object. Fixed to normalize datetimes via the original `json.dumps(..., default=str)` round-trip and then wrap the result in `Json(...)`.
- A handful of remaining `# type: ignore[...]` comments are left where the generated Prisma TypedDicts (`ArticleCreateWithoutRelationsInput`, `TopicSubscriptionInclude`, etc.) are stricter than the plain-dict-literal query-building style used everywhere in this codebase — each one has an inline comment explaining why, rather than being silently suppressed.
- Not done: wiring `ruff`/`mypy` into CI (no workflow currently runs them on push/PR) — straightforward to add to `.github/workflows/` if wanted.

## 🟡 Frontend — DONE

- [x] **F1** `[FE]` New `src/lib/api.ts` (`api.get`/`api.post`/`api.delete`) — one place for the base URL and `credentials: "include"`. Migrated all 13 call sites (turned out to be more than the ~10 estimated) across `page.tsx`, `history/page.tsx`, `subscriptions/page.tsx`, `dashboard/[id]/page.tsx`, `IntelligenceLayer.tsx`. `tsc --noEmit` passes.
- [x] **F2** `[FE]` Removed the stray `console.log("Bias data:", data)`.
- [x] **F3** `[FE]` README's Project Structure now lists the real routes (`history/`, `subscriptions/`, `login/` — `vault/` never existed, it's just the UI's label for `history/`) and the broken `docs/dashboard-preview.png` reference is gone (there's no `docs/` folder).
- [x] **F4** `[BE]`+`[FE]` Added `Search.phase2Status` (`pending`/`processing`/`complete`/`failed`, set by `pipeline.py` at each stage transition) and surfaced it in `get_search_intelligence()`'s response as `status`. `IntelligenceLayer.tsx` now polls only while status is `pending`/`processing` and stops the interval the moment it sees `complete`/`failed`; demo snapshots (static, no live pipeline) fetch once and never poll at all. **Needs one manual step** — see below.

**Manual step required (schema change, no access to your live DB from here):** run `prisma generate` (already done in this repo checkout) then `prisma db push` (or your normal deploy) to add the `phase2Status` column. It's an additive column with a default (`"pending"`) — no data cleanup needed first, unlike D1.

**Noted but not fixed:** `npx eslint` on the frontend surfaces 44 pre-existing issues (mostly `any` types, a few unescaped-JSX-quote and React-effect warnings) — none introduced by this pass, all in code that predates it. Wasn't part of F1–F4 and wasn't asked for; flagging it the same way CI-wiring for ruff/mypy was flagged on the backend, as an available next step rather than doing it unprompted.

## 🟡 Testing & Docs

- [x] **T1** `[BE]` Built a real, passing, exhaustive-scope pytest suite — **177 tests**, both README-named files included with content that matches their descriptions (see README's Testing section for the honest scoping note on `test_grounding.py`). Covers: the session-auth dependency + every route's enforcement of it (security-critical, tests actively assert 401/403/404 behavior), the claim quality gate + its Q1 word-boundary regressions, embedding-based dedup thresholds and cluster cohesion (real sentence-transformers model), the NLI cross-encoder for contradiction/grounding (real model), source-reliability/bias registries, narrative fallbacks, `get_results`/`get_search_intelligence`, the `/search` pipeline orchestration and Phase 2 status transitions, LLM response caching, domain-name parsing, JSON repair, and the Json-encoding regression in `create_demo_snapshot.py`. `ruff`/`mypy` both stay clean throughout.
  **Found and fixed 3 real, previously-undetected bugs while writing it** — see below.
  **Update — now closed:** `app/main.py`, `app/celery_app.py`, `app/tasks/snapshot_task.py`, and the remaining `app/utils/` scripts (`cleanup_orphaned_evidence.py`, `create_vector_index.py`, `reset_claim_graph.py`, `cohesion_analysis.py`) all have real tests now too. **213 tests total.** Every hand-written `.py` file in `app/` has at least one test file; only the vendored, generated `app/prisma_client/` is excluded (by design — see `pyproject.toml`'s mypy override). Three more small testability extractions happened along the way (`parse_allowed_origins` in `main.py`, `build_redis_url` in `celery_app.py`), same pattern as `normalize_domains` and `classify_nli_relationship` above.
- [x] **T2** `[BE]` Removed the dead `.gitignore` rule instead of the folder actually being ignored — `learning_notes/` is tracked in git regardless, so left a comment explaining why rather than re-adding a rule that would just be dead again.

## 🟢 DevOps — DONE

- [x] **V1** `[BE]` Removed `vercel.json` — confirmed nothing else in the repo referenced it (the two "vercel" mentions elsewhere are just the *frontend's* Vercel URL). Dockerfile + HF Spaces + the GitHub Action pushing there are the only real, actively-documented deploy path.

---
**Suggested order:** Security & Auth → Data Integrity → Architecture → everything else. See the [full audit](https://claude.ai/code/artifact/1a32c5de-9e38-4d5d-a473-73c56562a652) for rationale, code excerpts, and effort estimates per item.
