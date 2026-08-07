"""
One-time maintenance: add an approximate-nearest-neighbor index on
Claim.embedding.

Why this exists: Claim.embedding is `vector(384)` with NO index (confirmed
absent from the only migration file — see AUDIT_TASKS.md D2). Every claim
insertion runs a cross-article dedup query (app/services/extraction.py)
that orders the ENTIRE claim table by cosine distance to the new
embedding — an unindexed sequential scan that gets slower, linearly,
forever, as the table grows. This never shows up at demo scale; it will
show up eventually.

Prisma's schema DSL doesn't have first-class support for pgvector's
HNSW/IVFFlat index types, so this runs as raw SQL rather than a
schema.prisma change + `prisma db push` (which is why it's a script here
and not a migration file — the project currently manages its schema via
`db push`, which can't express this index type at all).

Uses HNSW (better default than IVFFlat for most workloads — no `lists`
tuning parameter to get wrong, and no periodic rebuild-as-the-table-grows
requirement). Requires pgvector >= 0.5.0; if your Postgres instance
predates that, switch to the commented IVFFLAT statement instead.

This takes a brief exclusive lock on the claim table while building (fine
for a one-time run; the table is not large yet). If it ever needs to be
rebuilt on a much larger table without blocking writes, run the
`CONCURRENTLY` variant by hand via psql instead of through this script —
CREATE INDEX CONCURRENTLY cannot run inside the kind of single-statement
call this script uses.

Usage:
    python -m app.utils.create_vector_index
"""

import asyncio
import logging

from app.db import prisma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# `<=>` (used throughout extraction.py/clustering.py) is pgvector's cosine
# distance operator — vector_cosine_ops is the matching index opclass.
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS claim_embedding_hnsw_idx
ON "claim" USING hnsw (embedding vector_cosine_ops)
"""

# Fallback for pgvector < 0.5.0 (no HNSW support). `lists` should be
# roughly sqrt(row_count) — 100 is a reasonable starting point for a table
# in the thousands-to-low-tens-of-thousands range; re-tune (and rebuild)
# once the table is much larger.
# CREATE_INDEX_SQL = """
# CREATE INDEX IF NOT EXISTS claim_embedding_ivfflat_idx
# ON "claim" USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
# """


async def main():
    await prisma.connect()
    try:
        logger.info("Creating HNSW index on claim.embedding (this may take a moment)...")
        await prisma.execute_raw(CREATE_INDEX_SQL)
        logger.info("Done. Cross-article dedup lookups should no longer full-scan the claim table.")
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
