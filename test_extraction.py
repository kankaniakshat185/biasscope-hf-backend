import asyncio
from app.prisma_client import Prisma
from app.services.extraction import process_and_store_claims

async def main():
    prisma = Prisma()
    await prisma.connect()
    
    # get one article
    article = await prisma.article.find_first(where={"content": {"not": None}})
    if not article:
        print("No article found")
        return
        
    print(f"Processing article: {article.title}")
    
    try:
        res = await process_and_store_claims(
            prisma,
            article.id,
            article.content,
            article.source,
            article.url,
            article.publishedAt,
            "elon musk",
            article.title
        )
        print("Inserted claim IDs:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
