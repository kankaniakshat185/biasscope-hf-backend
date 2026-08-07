"""
Development / admin infrastructure — debug endpoints.

Everything in this router does direct, unguarded data-plane operations
(including wiping the claim graph and the LLM cache). It is disabled by
default and only mounted when ENABLE_DEBUG_ROUTES=1 is set, and even then
requires a valid logged-in session — see app/deps/auth.py.

This does not implement role-based admin access (the User model has no
`role` column yet); any authenticated user can hit these when the flag is
on. Treat ENABLE_DEBUG_ROUTES as a "trusted single-operator environment
only" switch (local dev, or a locked-down staging instance) — it should
never be set in a public-facing deployment.
"""

from fastapi import APIRouter, BackgroundTasks, Depends

from ..db import prisma
from ..deps.auth import get_current_user_id, require_debug_enabled
from ..services.clustering import run_claim_clustering, run_event_detection
from ..services.extraction import process_and_store_claims

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
    dependencies=[Depends(require_debug_enabled), Depends(get_current_user_id)],
)


@router.get("/clusters")
async def debug_clusters():
    """Feature 5: Inspect all clusters without generating reports."""
    clusters = await prisma.claimcluster.find_many(
        include={"claims": {"include": {"evidence": True}}, "event": True},
        order={"id": "desc"},
    )
    result = []
    for cl in clusters:
        all_evidence = []
        for c in cl.claims:
            all_evidence.extend(c.evidence)
        sources = list(set(e.source for e in all_evidence))
        result.append({
            "cluster_id": cl.id,
            "title": cl.title,
            "canonicalClaim": cl.canonicalClaim,
            "consensusScore": cl.consensusScore,
            "claim_count": len(cl.claims),
            "evidence_count": len(all_evidence),
            "source_count": len(sources),
            "sources": sources,
            "claims": [c.canonicalClaim for c in cl.claims],
            "event_id": cl.eventId,
            "event_title": cl.event.title if cl.event else None,
        })
    return {"total": len(result), "clusters": result}


@router.get("/events")
async def debug_events():
    """Feature 6: Inspect all events without generating reports."""
    events = await prisma.event.find_many(
        include={"claimClusters": {"include": {"claims": {"include": {"evidence": True}}}}},
        order={"importanceScore": "desc"},
    )
    result = []
    for ev in events:
        total_claims = 0
        total_evidence = 0
        all_sources = set()
        for cl in ev.claimClusters:
            total_claims += len(cl.claims)
            for c in cl.claims:
                total_evidence += len(c.evidence)
                for e in c.evidence:
                    all_sources.add(e.source)
        result.append({
            "event_id": ev.id,
            "title": ev.title,
            "description": ev.description,
            "importance_score": ev.importanceScore,
            "canonical_claim": ev.claimClusters[0].canonicalClaim if ev.claimClusters else None,
            "cluster_count": len(ev.claimClusters),
            "claim_count": total_claims,
            "evidence_count": total_evidence,
            "source_count": len(all_sources),
            "sources": list(all_sources),
        })
    return {"total": len(result), "events": result}


@router.get("/llm-usage")
async def debug_llm_usage():
    """Feature 7: LLM usage analytics dashboard."""
    usage = await prisma.llmusage.find_many(order={"createdAt": "desc"})

    stages = {}
    total_cached = 0
    total_calls = 0
    for u in usage:
        stage = u.stage
        if stage not in stages:
            stages[stage] = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        stages[stage]["calls"] += 1
        if u.cached:
            stages[stage]["cached"] += 1
            total_cached += 1
        else:
            stages[stage]["prompt_tokens"] += u.promptTokens or 0
            stages[stage]["completion_tokens"] += u.completionTokens or 0
        total_calls += 1

    cache_hit_rate = (total_cached / max(total_calls, 1)) * 100

    return {
        "total_calls": total_calls,
        "total_cached": total_cached,
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "stages": stages,
    }


@router.get("/cache-stats")
async def debug_cache_stats():
    """Feature 1: View cache contents."""
    caches = await prisma.llmcache.find_many(order={"createdAt": "desc"})
    return {
        "total_cached_prompts": len(caches),
        "by_stage": {},
        "entries": [
            {
                "stage": c.stage,
                "model": c.model,
                "hash": c.promptHash[:12] + "...",
                "response_length": len(c.response),
                "created": c.createdAt,
            }
            for c in caches[:50]
        ],
    }


@router.post("/rerun-clustering")
async def debug_rerun_clustering(background_tasks: BackgroundTasks):
    """Feature 2: Rerun ONLY clustering + events (zero extraction cost). Runs in background."""
    await prisma.query_raw('UPDATE "claim" SET "clusterId" = NULL')
    await prisma.query_raw('DELETE FROM "claim_cluster"')
    await prisma.query_raw('DELETE FROM "event"')

    async def _run():
        try:
            await run_claim_clustering(prisma)
            await run_event_detection(prisma)
            print("Background rerun-clustering complete.")
        except Exception as e:
            import traceback
            print(f"Rerun-clustering error: {e}")
            traceback.print_exc()

    background_tasks.add_task(_run)
    return {"message": "Clustering rerun started in background. Check /debug/events after ~60s."}


@router.post("/rerun-events")
async def debug_rerun_events(background_tasks: BackgroundTasks):
    """Feature 2: Rerun ONLY event detection (zero extraction + clustering cost)."""
    await prisma.query_raw('UPDATE "claim_cluster" SET "eventId" = NULL')
    await prisma.query_raw('DELETE FROM "event"')

    async def _run():
        try:
            await run_event_detection(prisma)
            print("Background rerun-events complete.")
        except Exception as e:
            print(f"Rerun-events error: {e}")

    background_tasks.add_task(_run)
    return {"message": "Event rerun started in background. Check /debug/events after ~30s."}


@router.post("/clear-cache")
async def debug_clear_cache():
    """Clear the LLM response cache."""
    await prisma.query_raw('DELETE FROM "llm_cache"')
    await prisma.query_raw('DELETE FROM "llm_usage"')
    return {"message": "LLM cache and usage analytics cleared."}


@router.post("/reset-phase2")
async def debug_reset_phase2():
    """Wipe ALL Phase 2 data: claims, evidence, clusters, events. Use before re-extraction."""
    await prisma.query_raw('UPDATE "claim" SET "clusterId" = NULL')
    await prisma.query_raw('DELETE FROM "evidence"')
    await prisma.query_raw('DELETE FROM "claim"')
    await prisma.query_raw('DELETE FROM "claim_cluster"')
    await prisma.query_raw('DELETE FROM "event"')
    return {"message": "All Phase 2 data wiped. Run a search or /debug/rerun-full to re-extract."}


@router.post("/rerun-full")
async def debug_rerun_full(background_tasks: BackgroundTasks):
    """Re-extract claims from ALL existing articles, then cluster + detect events."""
    searches = await prisma.search.find_many(order={"createdAt": "desc"}, take=1)
    if not searches:
        return {"message": "No searches found."}

    search_id = searches[0].id
    query = searches[0].query

    async def _run():
        try:
            articles = await prisma.article.find_many(where={"searchId": search_id})
            print(f"Re-extracting from {len(articles)} articles for query='{query}'...")
            for art in articles:
                if art.content:
                    await process_and_store_claims(
                        prisma, art.id, art.content, art.source, art.url,
                        art.publishedAt, query, art.title,
                    )
            print("Re-extraction complete. Starting clustering...")
            await run_claim_clustering(prisma)
            print("Clustering complete. Starting event detection...")
            await run_event_detection(prisma)
            print("Full pipeline rerun complete.")
        except Exception as e:
            import traceback
            print(f"Rerun-full error: {e}")
            traceback.print_exc()

    background_tasks.add_task(_run)
    return {"message": f"Full rerun started for query='{query}', {search_id}. Check /debug/status."}


@router.get("/run-one")
async def debug_run_one():
    """Test extraction on a single article synchronously to catch errors."""
    import traceback
    try:
        article = await prisma.article.find_first(where={"content": {"not": None}})
        if not article:
            return {"error": "No article found"}

        claims = await process_and_store_claims(
            prisma, article.id, article.content, article.source, article.url,
            article.publishedAt, "elon musk", article.title
        )
        return {"success": True, "claims": claims}
    except Exception as e:
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}


@router.get("/status")
async def debug_status():
    """Quick status check — how many clusters/events exist right now."""
    clusters = await prisma.query_raw('SELECT COUNT(*) as cnt FROM "claim_cluster"')
    events = await prisma.query_raw('SELECT COUNT(*) as cnt FROM "event"')
    claims = await prisma.query_raw('SELECT COUNT(*) as cnt FROM "claim"')
    unclustered = await prisma.query_raw('SELECT COUNT(*) as cnt FROM "claim" WHERE "clusterId" IS NULL')
    return {
        "claims": claims[0]["cnt"] if claims else 0,
        "unclustered_claims": unclustered[0]["cnt"] if unclustered else 0,
        "clusters": clusters[0]["cnt"] if clusters else 0,
        "events": events[0]["cnt"] if events else 0,
    }


@router.get("/cluster-quality")
async def debug_cluster_quality():
    """Part 7: Per-cluster quality diagnostics."""
    clusters = await prisma.claimcluster.find_many(
        include={"claims": {"include": {"evidence": True}}},
        order={"id": "asc"},
    )

    diagnostics = []
    for cluster in clusters:
        claim_count = len(cluster.claims) if cluster.claims else 0
        all_evidence = []
        for c in (cluster.claims or []):
            all_evidence.extend(c.evidence)

        evidence_count = len(all_evidence)
        sources = set(e.source for e in all_evidence)
        source_count = len(sources)
        consensus = cluster.consensusScore or 0.0

        noise_score = 0.0
        if source_count <= 1 and evidence_count <= 2:
            noise_score += 0.5
        if claim_count <= 1:
            noise_score += 0.3
        if consensus < 0.2:
            noise_score += 0.2
        noise_score = min(noise_score, 1.0)

        is_multi_source = source_count >= 2
        is_substantial_single = claim_count >= 3 and evidence_count >= 4
        has_minimum_claims = claim_count >= 2
        has_minimum_evidence = evidence_count >= 2
        event_eligible = has_minimum_claims and has_minimum_evidence and (is_multi_source or is_substantial_single)

        diagnostics.append({
            "cluster_id": cluster.id,
            "title": cluster.title,
            "canonical_claim": cluster.canonicalClaim,
            "claim_count": claim_count,
            "evidence_count": evidence_count,
            "source_count": source_count,
            "sources": list(sources),
            "consensus_score": round(consensus, 3),
            "noise_score": round(noise_score, 3),
            "event_eligible": event_eligible,
            "event_id": cluster.eventId,
        })

    diagnostics.sort(key=lambda d: (-d["source_count"], d["noise_score"]))

    total_eligible = sum(1 for d in diagnostics if d["event_eligible"])
    total_noisy = sum(1 for d in diagnostics if d["noise_score"] >= 0.5)

    return {
        "total_clusters": len(diagnostics),
        "event_eligible": total_eligible,
        "noisy_clusters": total_noisy,
        "clusters": diagnostics,
    }
