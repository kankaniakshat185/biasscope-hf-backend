"""
FastAPI app entry point.

This module is intentionally just wiring: app construction, middleware,
and router registration. Route handlers live in app/routers/, and the
business logic they call lives in app/services/ — see each module's
docstring. (Previously this single 828-line file mixed routing, pipeline
orchestration, and admin/debug tooling together; see AUDIT_TASKS.md A1.)
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import prisma
from .routers import chat, history, results, search, subscriptions
from .routers.debug import router as debug_router

# Application code logs via `logging.getLogger(__name__)` throughout
# app/services and app/routers (this used to be a mix of print() and
# logging with nothing configuring the root logger — see AUDIT_TASKS.md
# Q2). Without this, logger.info() calls are silently dropped: Python's
# root logger defaults to WARNING with no handler, and uvicorn configures
# its OWN loggers (uvicorn.*) but not the application's. HF Spaces/most
# container platforms just capture stdout, so a plain basicConfig is
# enough to make these show up in the deployment's logs.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Biascope API")

_DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,https://biasscope-app-frontend.vercel.app"


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Comma-separated list of allowed frontend origins. Never pair "*"
    with allow_credentials=True — browsers reject that combination for
    credentialed requests, and it signals an unintentionally open policy
    for an API that now reads session cookies. Extracted for direct
    testability (see tests/test_main.py)."""
    return [origin.strip() for origin in (raw or _DEFAULT_ALLOWED_ORIGINS).split(",") if origin.strip()]


ALLOWED_ORIGINS = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(results.router)
app.include_router(subscriptions.router)
app.include_router(history.router)
app.include_router(chat.router)
app.include_router(debug_router)


@app.on_event("startup")
async def startup():
    # Schema sync intentionally does NOT happen here anymore — running
    # `prisma db push` as a side effect of every app-process boot silently
    # swallowed failures and could race with the checked-in migration
    # history. It now runs once, loudly, as part of the container's start
    # command (see Dockerfile) so a failed sync stops the deploy instead of
    # limping into an inconsistent schema.
    await prisma.connect()


@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()


@app.get("/")
def read_root():
    return {"status": "ok", "service": "Biascope Backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
