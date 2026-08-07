"""Topic subscription management — all routes operate on the authenticated
caller's own subscriptions; identity comes from the session, never from a
client-supplied user id (see app/deps/auth.py)."""

from fastapi import APIRouter, Body, Depends

from ..db import prisma
from ..deps.auth import get_current_user_id

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.post("")
async def subscribe_topic(topic: str = Body(...), user_id: str = Depends(get_current_user_id)):
    """Subscribe the authenticated user to a topic for weekly longitudinal tracking."""
    existing = await prisma.topicsubscription.find_first(
        where={"userId": user_id, "topic": topic.lower()}
    )
    if existing:
        if not existing.isActive:
            await prisma.topicsubscription.update(where={"id": existing.id}, data={"isActive": True})
        return existing

    return await prisma.topicsubscription.create(
        data={"userId": user_id, "topic": topic.lower()}
    )


@router.get("")
async def get_subscriptions(user_id: str = Depends(get_current_user_id)):
    """Get all active subscriptions for the authenticated user."""
    return await prisma.topicsubscription.find_many(
        where={"userId": user_id, "isActive": True},
        include={"snapshots": {"orderBy": {"createdAt": "desc"}, "take": 5}},
    )


@router.delete("")
async def unsubscribe_topic(topic: str, user_id: str = Depends(get_current_user_id)):
    """Deactivate a topic subscription for the authenticated user."""
    sub = await prisma.topicsubscription.find_first(
        where={"userId": user_id, "topic": topic.lower()}
    )
    if sub:
        await prisma.topicsubscription.update(
            where={"id": sub.id},
            data={"isActive": False},
        )
    return {"status": "success"}
