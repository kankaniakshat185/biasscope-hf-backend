"""Topic subscription management — all routes operate on the authenticated
caller's own subscriptions; identity comes from the session, never from a
client-supplied user id (see app/deps/auth.py)."""

from fastapi import APIRouter, Body, Depends

from ..db import prisma
from ..deps.auth import get_current_user_id

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.post("")
async def subscribe_topic(
    # embed=True: with `userId` removed as a sibling Body(...) field (S2 —
    # identity now comes from the session, not the client), `topic` became
    # the ONLY Body param. FastAPI's default for a single scalar Body
    # param is to expect the raw value as the entire request body, not
    # `{"topic": "..."}` — which silently broke this endpoint against the
    # frontend (still sends the wrapped shape) until a test caught it.
    topic: str = Body(..., embed=True),
    user_id: str = Depends(get_current_user_id),
):
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
        # Same generated-TypedDict-vs-dict-literal friction as elsewhere
        # (see pipeline.py) — the mixed dict/int values here make the
        # inferred type too loose to match TopicSubscriptionInclude exactly.
        include={"snapshots": {"orderBy": {"createdAt": "desc"}, "take": 5}},  # type: ignore[arg-type]
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
