import asyncio
import logging
from app.prisma_client import Prisma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def analyze_cohesion():
    prisma = Prisma()
    await prisma.connect()

    clusters = await prisma.claimcluster.find_many(
        include={
            "claims": {
                "include": {
                    "evidence": True
                }
            },
            "event": True
        },
        order={"cohesionScore": "desc"}
    )

    print(f"{'Cluster Title':<40} | {'Claims':<6} | {'Sources':<7} | {'Cohesion':<8} | {'Event?'}")
    print("-" * 80)

    for cluster in clusters:
        title = (cluster.title[:37] + "...") if len(cluster.title) > 37 else cluster.title
        claim_count = len(cluster.claims)
        
        sources = set()
        for claim in cluster.claims:
            for ev in claim.evidence:
                sources.add(ev.source)
        source_count = len(sources)
        
        cohesion = cluster.cohesionScore if cluster.cohesionScore is not None else 0.0
        accepted = "Yes" if cluster.eventId else "No"
        
        print(f"{title:<40} | {claim_count:<6} | {source_count:<7} | {cohesion:<8.3f} | {accepted}")

    await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(analyze_cohesion())
