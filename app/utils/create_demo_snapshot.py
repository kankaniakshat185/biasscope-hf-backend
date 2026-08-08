import asyncio
import json
import logging
from datetime import datetime

from app.prisma_client import Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_snapshot_json(demo_data: dict) -> Json:
    """Normalizes a demo-snapshot dict into something Prisma's `Json`
    wrapper can actually serialize, then wraps it.

    Extracted for direct testability — see tests/utils/test_create_demo_snapshot.py
    for the regression test covering the bug this fixed: passing
    json.dumps(...) (a plain string) here instead of Json(...) made the
    generated client's default string serializer double-encode the data,
    so DemoSnapshot.data ended up holding a JSON-encoded STRING rather
    than a JSON OBJECT — /demo/{topic} would have handed the frontend
    escaped JSON text instead of a parsed object.

    Still round-trips through json.dumps/loads first (`default=str`) to
    normalize embedded datetime objects — Prisma's own Json serializer
    does a plain json.dumps with no `default=`, so it can't handle those
    directly.
    """
    normalized = json.loads(json.dumps(demo_data, default=str))
    return Json(normalized)


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

        snapshot_json = build_snapshot_json(demo_data)

        # Save to demo snapshot table
        await prisma.demosnapshot.upsert(
            where={"topic": topic},
            data={
                "create": {
                    "topic": topic,
                    "data": snapshot_json
                },
                "update": {
                    "data": snapshot_json
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
