"""
One-time cleanup: delete Evidence rows whose articleId no longer points at
an existing Article.

Why this exists: Evidence.articleId never had a foreign key constraint
(see AUDIT_TASKS.md D1), so deleting a Search (which cascades to its
Articles) silently left orphaned Evidence rows behind. Adding the FK
constraint now — `article Article @relation(..., onDelete: Cascade)` in
schema.prisma — will fail with a constraint-violation error via `prisma db
push` if any orphans currently exist. Run this script once, THEN push the
schema change.

Usage:
    python -m app.utils.cleanup_orphaned_evidence          # dry run, reports count only
    python -m app.utils.cleanup_orphaned_evidence --delete # actually deletes them
"""

import asyncio
import logging
import sys

from app.db import prisma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def find_orphaned_evidence_count() -> int:
    result = await prisma.query_raw(
        """
        SELECT COUNT(*) as cnt
        FROM "evidence" e
        WHERE NOT EXISTS (
            SELECT 1 FROM "article" a WHERE a.id = e."articleId"
        )
        """
    )
    return result[0]["cnt"] if result else 0


async def delete_orphaned_evidence() -> int:
    result = await prisma.query_raw(
        """
        DELETE FROM "evidence" e
        WHERE NOT EXISTS (
            SELECT 1 FROM "article" a WHERE a.id = e."articleId"
        )
        RETURNING e.id
        """
    )
    return len(result) if result else 0


async def main():
    do_delete = "--delete" in sys.argv

    await prisma.connect()
    try:
        count = await find_orphaned_evidence_count()
        if count == 0:
            logger.info("No orphaned evidence rows found. Safe to add the FK constraint.")
            return

        if not do_delete:
            logger.warning(
                f"Found {count} orphaned evidence row(s) (articleId points at a "
                f"deleted article). Re-run with --delete to remove them. "
                f"The FK constraint in schema.prisma cannot be applied until this is 0."
            )
            return

        deleted = await delete_orphaned_evidence()
        logger.info(f"Deleted {deleted} orphaned evidence row(s).")
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
