"""
Router-test helpers.

Deliberately builds a bare FastAPI() app and mounts only the router under
test, rather than importing app.main's real `app` object — main.py
registers an @app.on_event("startup") that calls `await prisma.connect()`
against whatever DATABASE_URL is in the environment. This repo's local
.env has a REAL production connection string in it; a test run must never
risk opening that. Routers themselves have no lifespan events, so mounting
just the router sidesteps the whole problem.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)
