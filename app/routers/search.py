"""The main /search endpoint — kicks off the synchronous analysis pipeline
and schedules the Phase 2 (claims/clustering/events) background job."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends

from ..deps.auth import get_optional_user_id
from ..services.pipeline import run_phase2_pipeline, run_search_pipeline

router = APIRouter(tags=["search"])


@router.post("/search")
async def create_search(
    query: str = Body(...),
    category: str = Body(...),
    domains: str = Body(None),
    exclude_domains: str = Body(None),
    fromDate: str = Body(None),
    toDate: str = Body(None),
    background_tasks: BackgroundTasks = None,
    # Anonymous search is allowed (Search.userId is nullable), but if the
    # caller IS logged in, their id comes from the verified session — never
    # from a client-supplied body field.
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    result = await run_search_pipeline(
        query=query,
        category=category,
        domains=domains,
        exclude_domains=exclude_domains,
        from_date=fromDate,
        to_date=toDate,
        user_id=user_id,
    )

    if background_tasks:
        background_tasks.add_task(run_phase2_pipeline, result["search_id"])

    return result
