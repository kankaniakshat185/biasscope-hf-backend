import asyncio
import json
import logging
from datetime import datetime
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    topic = "elon musk"
    search_id = "0d5391aa-59ac-442a-8876-484d2db95a3e"
    
    from app.main import get_results, get_search_intelligence, prisma

    await prisma.connect()

    logger.info(f"Generating demo snapshot for topic '{topic}' using search ID {search_id}")

    try:
        base_record = await get_results(search_id)
        intel_data = await get_search_intelligence(search_id)
        
        # Pydantic serialization
        if hasattr(base_record, "model_dump"):
            base_data = base_record.model_dump(mode="json")
        else:
            base_data = json.loads(base_record.json())
            
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
