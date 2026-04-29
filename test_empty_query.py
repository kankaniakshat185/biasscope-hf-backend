import asyncio
from app.services.ingestion import ingest_articles

async def main():
    print("Testing with domain and category but NO query...")
    try:
        res = await ingest_articles(query="", category="Technology", domains="wsj.com")
        print("Results length:", len(res))
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
