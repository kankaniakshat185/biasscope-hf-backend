"""Read-only routes for fetching a search's results and derived intelligence."""

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
    return await get_results(search_id)


@router.get("/results/{search_id}/intelligence")
async def read_search_intelligence(search_id: str):
    return await get_search_intelligence(search_id)
