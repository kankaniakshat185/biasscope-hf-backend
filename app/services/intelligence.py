"""
Read-side intelligence queries — fetching a search's results and its
derived claim/cluster/event graph.

Pulled out of app/main.py so that Celery tasks and one-off scripts can
reuse this logic without importing the FastAPI app module itself (which
used to pull in app construction, CORS middleware, and route registration
as side effects of wanting one function).
"""

from fastapi import HTTPException

from ..db import prisma
from .nlp import get_source_reliability


async def get_results(search_id: str) -> dict:
    """Fetch a search's stored articles + insights, with each article's
    static source-reliability tier attached (Issue A3 — this used to be
    computed independently, and inconsistently, in the frontend)."""
    search_record = await prisma.search.find_unique(
        where={"id": search_id},
        include={"articles": True, "insights": True},
    )
    if not search_record:
        raise HTTPException(status_code=404, detail="Search not found")

    data = search_record.model_dump(mode="json")
    for article in data.get("articles", []):
        score, tier = get_source_reliability(article.get("source", ""))
        article["reliabilityScore"] = score
        article["reliabilityTier"] = tier
    return data


async def get_search_intelligence(search_id: str) -> dict:
    """Build the claim/cluster/event intelligence graph for a search."""
    articles = await prisma.article.find_many(where={"searchId": search_id})
    article_ids = [a.id for a in articles]

    if not article_ids:
        return {"events": [], "clusters": [], "claims": [], "metrics": {}}

    evidence_records = await prisma.evidence.find_many(
        where={"articleId": {"in": article_ids}},
        include={"claim": True},
    )

    claim_ids = list(set([e.claimId for e in evidence_records if e.claimId]))

    claims = await prisma.claim.find_many(
        where={"id": {"in": claim_ids}},
        include={
            "evidence": True,
            "cluster": {
                "include": {
                    "event": True,
                }
            },
        },
    )

    # ── Build structured output ──
    # Key fix: canonical claim lives on CLUSTER, raw text on CLAIMS
    # Evidence aggregation happens at cluster level for consistency

    formatted_claims = []
    clusters_map = {}
    events_map = {}

    for c in claims:
        # Each claim keeps its ORIGINAL raw text
        claim_evidence = [
            {"sentence": e.sentence, "source": e.source, "publishedAt": e.publishedAt, "url": e.url}
            for e in c.evidence
        ]
        claim_sources = list(set(e.source for e in c.evidence))

        fc = {
            "id": c.id,
            "canonicalClaim": c.canonicalClaim,  # original raw text
            "claimType": getattr(c, 'claimType', 'EVENT') or 'EVENT',
            "qualityScore": getattr(c, 'qualityScore', 0) or 0,
            "confidence": c.confidence,
            "evidenceCount": len(c.evidence),
            "sources": claim_sources,
            "evidence": claim_evidence,
            "clusterId": c.clusterId,
        }
        formatted_claims.append(fc)

        # Build cluster — canonical claim comes from CLUSTER, not from claim
        if c.cluster:
            cid = c.cluster.id
            if cid not in clusters_map:
                clusters_map[cid] = {
                    "id": cid,
                    "title": c.cluster.title,
                    "canonicalClaim": getattr(c.cluster, 'canonicalClaim', '') or '',
                    "consensusScore": getattr(c.cluster, 'consensusScore', 0) or 0,
                    "eventId": c.cluster.eventId,
                    "rawClaims": [],       # original claim texts (not canonical)
                    "allEvidence": [],      # all evidence across all claims
                    "sources": set(),
                    "claimCount": 0,
                }
            # Store the RAW claim text (not canonical) as supporting claim
            if c.canonicalClaim not in [rc["text"] for rc in clusters_map[cid]["rawClaims"]]:
                clusters_map[cid]["rawClaims"].append({"text": c.canonicalClaim, "id": c.id})
            clusters_map[cid]["claimCount"] += 1
            # Aggregate ALL evidence at cluster level
            clusters_map[cid]["allEvidence"].extend(claim_evidence)
            for s in claim_sources:
                clusters_map[cid]["sources"].add(s)

            # Build event
            if c.cluster.event:
                eid = c.cluster.event.id
                if eid not in events_map:
                    events_map[eid] = {
                        "id": eid,
                        "title": c.cluster.event.title,
                        "description": getattr(c.cluster.event, 'description', '') or '',
                        "importanceScore": getattr(c.cluster.event, 'importanceScore', 0) or 0,
                        "canonicalClaim": getattr(c.cluster, 'canonicalClaim', '') or '',
                        "clusters": [],
                        "claimCount": 0,
                        "evidenceCount": 0,
                        "allEvidence": [],
                        "sources": set(),
                    }
                if c.cluster.title not in events_map[eid]["clusters"]:
                    events_map[eid]["clusters"].append(c.cluster.title)
                events_map[eid]["claimCount"] += 1
                events_map[eid]["evidenceCount"] += len(claim_evidence)
                events_map[eid]["allEvidence"].extend(claim_evidence)
                for s in claim_sources:
                    events_map[eid]["sources"].add(s)

    # Format clusters
    formatted_clusters = []
    for cl in clusters_map.values():
        cl["sources"] = list(cl["sources"])
        cl["sourceCount"] = len(cl["sources"])
        cl["evidenceCount"] = len(cl["allEvidence"])
        # Deduplicate evidence by sentence text
        seen = set()
        unique_evidence = []
        for ev in cl["allEvidence"]:
            key = ev["sentence"][:100]
            if key not in seen:
                seen.add(key)
                unique_evidence.append(ev)
        cl["evidence"] = unique_evidence
        cl["claims"] = [rc["text"] for rc in cl["rawClaims"]]
        del cl["rawClaims"]
        del cl["allEvidence"]
        formatted_clusters.append(cl)

    # Format events
    formatted_events = []
    for ev in events_map.values():
        ev["sources"] = list(ev["sources"])
        ev["sourceCount"] = len(ev["sources"])
        # Deduplicate evidence
        seen = set()
        unique_evidence = []
        for e in ev["allEvidence"]:
            key = e["sentence"][:100]
            if key not in seen:
                seen.add(key)
                unique_evidence.append(e)
        ev["evidence"] = unique_evidence
        ev["evidenceCount"] = len(unique_evidence)
        del ev["allEvidence"]
        formatted_events.append(ev)

    return {
        "metrics": {
            "articlesProcessed": len(article_ids),
            "claimsExtracted": len(evidence_records),
            "canonicalClaims": len(formatted_claims),
            "clusters": len(formatted_clusters),
            "events": len(formatted_events),
        },
        "claims": sorted(formatted_claims, key=lambda x: x["evidenceCount"], reverse=True),
        "clusters": sorted(formatted_clusters, key=lambda x: x["evidenceCount"], reverse=True),
        "events": sorted(formatted_events, key=lambda x: x.get("importanceScore", 0), reverse=True),
    }
