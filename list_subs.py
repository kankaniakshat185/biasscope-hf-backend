import asyncio
from app.prisma_client import Prisma, Json

async def main():
    p = Prisma()
    await p.connect()
    subs = await p.topicsubscription.find_many()
    
    import json
    from datetime import datetime, timedelta
    
    for sub in subs:
        print(f"Mocking data for: {sub.topic}")
        # Delete existing to prevent duplication errors
        await p.topicsnapshot.delete_many(where={"subscriptionId": sub.id})
        
        # Create week 1 (7 days ago)
        await p.topicsnapshot.create(
            data={
                "subscriptionId": sub.id,
                "topic": sub.topic,
                "createdAt": datetime.utcnow() - timedelta(days=7),
                "articleCount": 15,
                "claimCount": 24,
                "eventCount": 4,
                "polarizationIndex": 0.40,
                "biasDistribution": Json({"LEFT": 4, "CENTER": 7, "RIGHT": 4}),
                "sourceDistribution": Json({"nytimes.com": 4, "wsj.com": 4, "reuters.com": 7})
            }
        )
        
        # Create week 2 (today)
        await p.topicsnapshot.create(
            data={
                "subscriptionId": sub.id,
                "topic": sub.topic,
                "createdAt": datetime.utcnow(),
                "articleCount": 20,
                "claimCount": 35,
                "eventCount": 6,
                "polarizationIndex": 0.55,
                "biasDistribution": Json({"LEFT": 8, "CENTER": 9, "RIGHT": 3}),
                "sourceDistribution": Json({"nytimes.com": 8, "wsj.com": 3, "reuters.com": 9})
            }
        )
        print("Done!")

    await p.disconnect()

asyncio.run(main())
