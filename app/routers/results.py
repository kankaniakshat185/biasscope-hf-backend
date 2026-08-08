"""Read-only routes for fetching a search's results and derived intelligence.

INTENTIONALLY UNAUTHENTICATED — this is a product decision, not an
oversight. A search's `search_id` is a UUIDv4 (unguessable) and doubles as
a shareable link: anyone with the URL — logged in or not, same account or
not — can view that search's report, same model as "anyone with the link
can view" document sharing. This was flagged as a potential IDOR in the
2026-08-08 audit (AUDIT_TASKS.md R2); the explicit call, confirmed with the
product owner, is to keep it link-shareable rather than lock it to the
creating account.

This does NOT make a user's search history public — `GET /history` (see
history.py) still requires a valid session and only ever returns the
caller's own searches. What's shareable is an individual report once you
already have its link, not the ability to discover or enumerate other
people's searches.

If this ever needs to become owner-only for a specific search (private
mode), the fix is `get_optional_user_id` + a check that
`search_record.userId in (None, user_id)` — matching the pattern already
used in `history.py`'s DELETE route — not a blanket auth requirement,
since anonymous search (Search.userId is nullable) must keep working.
"""

from fastapi import APIRouter, HTTPException

from ..db import prisma
from ..services.intelligence import get_results, get_search_intelligence

router = APIRouter(tags=["results"])


@router.get("/demo/{topic}")
async def get_demo_snapshot(topic: str):
    """Returns a fully precomputed intelligence report instantly for demo purposes."""
    snapshot = await prisma.demosnapshot.find_unique(where={"topic": topic.lower()})
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"No demo snapshot found for topic '{topic}'")
    return snapshot.data


@router.get("/results/{search_id}")
async def read_results(search_id: str):
    """No ownership check — see the module docstring for why this is
    intentional (shareable-by-link), not a missing S1-style auth gate."""
    return await get_results(search_id)


@router.get("/results/{search_id}/intelligence")
async def read_search_intelligence(search_id: str):
    """No ownership check — see the module docstring for why this is
    intentional (shareable-by-link), not a missing S1-style auth gate."""
    return await get_search_intelligence(search_id)
