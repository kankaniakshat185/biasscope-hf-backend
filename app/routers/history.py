"""Search history — always scoped to the authenticated caller."""

from fastapi import APIRouter, Depends, HTTPException

from ..db import prisma
from ..deps.auth import get_current_user_id

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
async def get_history(user_id: str = Depends(get_current_user_id)):
    """Retrieve past searches belonging to the authenticated user."""
    return await prisma.search.find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
        include={"insights": True},
    )


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
