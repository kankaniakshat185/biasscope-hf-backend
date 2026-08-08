"""The main /search endpoint — kicks off the synchronous analysis pipeline
and schedules the Phase 2 (claims/clustering/events) background job."""


from fastapi import APIRouter, BackgroundTasks, Body, Depends

from ..deps.auth import get_optional_user_id
from ..services.pipeline import run_phase2_pipeline, run_search_pipeline

router = APIRouter(tags=["search"])


@router.post("/search")
async def create_search(
    query: str = Body(...),
    # The frontend omits `category` from the JSON body entirely when the
    # user hasn't picked one (`category || undefined`, dropped by
    # JSON.stringify) — a required Body(...) here made every such request
    # fail with 422 before ingestion.py's "" == no-filter handling ever ran.
    category: str = Body(""),
    domains: str | None = Body(None),
    exclude_domains: str | None = Body(None),
    fromDate: str | None = Body(None),
    toDate: str | None = Body(None),
    # Must stay a BARE `BackgroundTasks` annotation, NOT `BackgroundTasks
    # | None` — FastAPI special-cases the exact bare type to auto-inject
    # an instance; wrapped in Optional, it instead tries (and fails) to
    # build a Pydantic field from it, crashing at route-registration time
    # (i.e. app startup). A prior mypy cleanup pass "fixed" this to satisfy
    # mypy's implicit-optional check and broke the route entirely — caught
    # by tests/routers/test_search_router.py, not by mypy or ruff.
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
    # Anonymous search is allowed (Search.userId is nullable), but if the
    # caller IS logged in, their id comes from the verified session — never
    # from a client-supplied body field.
    user_id: str | None = Depends(get_optional_user_id),
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
