"""Search history — always scoped to the authenticated caller."""

from fastapi import APIRouter, Depends, HTTPException, Query

from ..db import prisma
from ..deps.auth import get_current_user_id

router = APIRouter(prefix="/history", tags=["history"])

# This used to return every search a user had ever run, unbounded — fine
# for a demo account, not fine after months of real usage. Default limit
# is set high enough that it's a no-op for virtually every account today
# (nobody has 50+ searches yet); it exists to put a ceiling on the query
# rather than to change today's behavior.
DEFAULT_HISTORY_LIMIT = 50
MAX_HISTORY_LIMIT = 200


@router.get("")
async def get_history(
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Retrieve past searches belonging to the authenticated user, newest first."""
    total = await prisma.search.count(where={"userId": user_id})
    searches = await prisma.search.find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
        include={"insights": True},
        take=limit,
        skip=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "searches": searches}


@router.delete("/{search_id}")
async def delete_search(search_id: str, user_id: str = Depends(get_current_user_id)):
    search_record = await prisma.search.find_unique(where={"id": search_id})
    if not search_record:
        raise HTTPException(status_code=404, detail="Search not found")

    if search_record.userId != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to delete this search.",
        )

    await prisma.search.delete(where={"id": search_id})
    return {"message": "Search and all associated data permanently deleted."}
