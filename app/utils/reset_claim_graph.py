"""
Database Reset Utility

Wipes all intelligence graph data (claims, clusters, events, evidence)
while preserving user accounts and authentication state.

Usage:
    python -m app.utils.reset_claim_graph
"""

import asyncio
import logging
from app.prisma_client import Prisma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reset_claim_graph():
    """Delete all intelligence pipeline data for a clean test run."""
    prisma = Prisma()
    await prisma.connect()

    logger.info("Resetting claim graph...")

    try:
        await prisma.evidence.delete_many(where={})
        await prisma.contradictionpair.delete_many(where={})
        await prisma.consensusfact.delete_many(where={})
        await prisma.event.delete_many(where={})
        await prisma.claimcluster.delete_many(where={})
        await prisma.claim.delete_many(where={})
        await prisma.llmcache.delete_many(where={})
        await prisma.llmusage.delete_many(where={})
        logger.info("✅ Claim graph reset complete.")
    except Exception as e:
        logger.error(f"Reset failed: {e}")
    finally:
        await prisma.disconnect()


async def reset_all():
    """Delete ALL run data (articles, insights, searches) plus claim graph."""
    prisma = Prisma()
    await prisma.connect()

    logger.info("Full database reset...")

    try:
        await prisma.evidence.delete_many(where={})
        await prisma.contradictionpair.delete_many(where={})
        await prisma.consensusfact.delete_many(where={})
        await prisma.event.delete_many(where={})
        await prisma.claimcluster.delete_many(where={})
        await prisma.claim.delete_many(where={})
        await prisma.article.delete_many(where={})
        await prisma.insight.delete_many(where={})
        await prisma.search.delete_many(where={})
        await prisma.llmcache.delete_many(where={})
        await prisma.llmusage.delete_many(where={})
        logger.info("✅ Full database reset complete.")
    except Exception as e:
        logger.error(f"Reset failed: {e}")
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        asyncio.run(reset_all())
    else:
        asyncio.run(reset_claim_graph())
