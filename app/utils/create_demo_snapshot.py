import asyncio
import httpx
import json
import logging
from app.prisma_client import Prisma, Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_demo_snapshot(search_id: str, topic: str):
    prisma = Prisma()
    await prisma.connect()

    logger.info(f"Generating demo snapshot for topic '{topic}' using search ID {search_id}")

    try:
        # Base results
        base_record = await prisma.search.find_unique(
            where={"id": search_id},
            include={"articles": True, "insights": True}
        )
        base_data = base_record.model_dump(mode="json") if hasattr(base_record, "model_dump") else base_record.dict()

        # Intelligence data
        articles = await prisma.article.find_many(where={"searchId": search_id})
        article_ids = [a.id for a in articles]
        
        intel_data = {"events": [], "clusters": [], "claims": [], "metrics": {}}
        if article_ids:
            evidence_records = await prisma.evidence.find_many(
                where={"articleId": {"in": article_ids}},
                include={"claim": True}
            )
            claim_ids = list(set([e.claimId for e in evidence_records if e.claimId]))
            claims = await prisma.claim.find_many(
                where={"id": {"in": claim_ids}},
                include={
                    "evidence": True,
                    "cluster": {
                        "include": {
                            "event": True
                        }
                    }
                }
            )
            
            formatted_claims = []
            clusters_map = {}
            events_map = {}
            
            for c in claims:
                claim_evidence = [
                    {"sentence": e.sentence, "source": e.source, "publishedAt": str(e.publishedAt), "url": e.url}
                    for e in c.evidence
                ]
                claim_sources = list(set(e.source for e in c.evidence))
                
                fc = {
                    "id": c.id,
                    "canonicalClaim": c.canonicalClaim,
                    "claimType": c.claimType,
                    "qualityScore": c.qualityScore,
                    "confidence": c.confidence,
                    "sources": claim_sources,
                    "evidence": claim_evidence,
                    "clusterId": c.clusterId
                }
                formatted_claims.append(fc)
                
                if c.cluster:
                    clus = c.cluster
                    if clus.id not in clusters_map:
                        clusters_map[clus.id] = {
                            "id": clus.id,
                            "title": clus.title,
                            "canonicalClaim": clus.canonicalClaim,
                            "consensusScore": clus.consensusScore,
                            "cohesionScore": clus.cohesionScore,
                            "eventId": clus.eventId,
                            "claims": []
                        }
                    clusters_map[clus.id]["claims"].append(fc)
                    
                    if clus.event:
                        evt = clus.event
                        if evt.id not in events_map:
                            events_map[evt.id] = {
                                "id": evt.id,
                                "title": evt.title,
                                "description": evt.description,
                                "importanceScore": evt.importanceScore,
                                "clusters": []
                            }
                        if clus.id not in [cl["id"] for cl in events_map[evt.id]["clusters"]]:
                            events_map[evt.id]["clusters"].append(clusters_map[clus.id])
            
            intel_data = {
                "events": sorted(list(events_map.values()), key=lambda x: x.get("importanceScore", 0) or 0, reverse=True),
                "clusters": list(clusters_map.values()),
                "claims": formatted_claims,
                "metrics": {
                    "totalClaims": len(claims),
                    "totalClusters": len(clusters_map),
                    "totalEvents": len(events_map)
                }
            }

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    # Combine into a single frontend-ready payload
    demo_data = {
        "topic": topic,
        "search": base_data,
        "intelligence": intel_data
    }

    # Upsert snapshot
    existing = await prisma.demosnapshot.find_unique(where={"topic": topic.lower()})
    
    if existing:
        await prisma.demosnapshot.update(
            where={"id": existing.id},
            data={"data": Json(demo_data)}
        )
        logger.info(f"Updated existing demo snapshot for '{topic}'")
    else:
        await prisma.demosnapshot.create(
            data={
                "topic": topic.lower(),
                "data": Json(demo_data)
            }
        )
        logger.info(f"Created new demo snapshot for '{topic}'")

    await prisma.disconnect()

if __name__ == "__main__":
    # We use the search ID we just found
    asyncio.run(create_demo_snapshot("0d5391aa-59ac-442a-8876-484d2db95a3e", "elon musk"))
