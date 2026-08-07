import asyncio
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    topic = "elon musk"
    search_id = "0d5391aa-59ac-442a-8876-484d2db95a3e"

    # Import the service functions directly instead of from app.main — this
    # script used to import app.main just to reuse two functions, which
    # meant it transitively depended on FastAPI app construction and its
    # startup side effects. See AUDIT_TASKS.md A2.
    from app.db import prisma
    from app.services.intelligence import get_results, get_search_intelligence

    await prisma.connect()

    logger.info(f"Generating demo snapshot for topic '{topic}' using search ID {search_id}")

    try:
        # get_results now returns a plain dict (already JSON-serializable)
        # rather than a Prisma model instance — see app/services/intelligence.py.
        base_data = await get_results(search_id)
        intel_data = await get_search_intelligence(search_id)

        demo_data = {
            "id": f"demo-{search_id}",
            "topic": topic,
            "createdAt": datetime.utcnow().isoformat(),
            "search": base_data,
            "intelligence": intel_data
        }

        # Save to demo snapshot table
        await prisma.demosnapshot.upsert(
            where={"topic": topic},
            data={
                "create": {
                    "topic": topic,
                    "data": json.dumps(demo_data, default=str)
                },
                "update": {
                    "data": json.dumps(demo_data, default=str)
                }
            }
        )

        logger.info(f"Created new demo snapshot for '{topic}'")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
    finally:
        await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
