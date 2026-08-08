"""app/routers/search.py — auth is OPTIONAL here (anonymous search is a
supported product behavior, Search.userId is nullable), but when the
caller IS logged in, the id must come from the session, never a body
field. The actual pipeline logic is mocked here — it's covered directly
in tests/services/test_pipeline.py."""

from unittest.mock import AsyncMock

import pytest

from app.deps.auth import get_optional_user_id
from app.routers import search as search_module
from tests.routers.conftest import make_client


@pytest.fixture
def client():
    return make_client(search_module.router)


@pytest.fixture
def mocked_pipeline(monkeypatch):
    run_search = AsyncMock(return_value={"search_id": "s1", "message": "ok"})
    run_phase2 = AsyncMock()
    monkeypatch.setattr(search_module, "run_search_pipeline", run_search)
    monkeypatch.setattr(search_module, "run_phase2_pipeline", run_phase2)
    return run_search, run_phase2


def test_anonymous_search_is_allowed(client, mocked_pipeline, monkeypatch):
    from tests.fakes import FakePrisma
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())  # no session -> anonymous
    run_search, _ = mocked_pipeline

    res = client.post("/search", json={"query": "elon musk", "category": "Technology"})

    assert res.status_code == 200
    assert run_search.call_args.kwargs["user_id"] is None


def test_logged_in_search_uses_the_session_user_id_not_a_body_field(client, mocked_pipeline):
    client.app.dependency_overrides[get_optional_user_id] = lambda: "user-1"
    run_search, _ = mocked_pipeline

    res = client.post("/search", json={
        "query": "elon musk", "category": "Technology",
        # Even if a client tried to sneak a userId into the body, the
        # endpoint doesn't declare that field at all — Pydantic/FastAPI
        # would just ignore it, not use it.
        "userId": "someone-elses-id",
    })

    assert res.status_code == 200
    assert run_search.call_args.kwargs["user_id"] == "user-1"


def test_schedules_the_phase2_background_pipeline(client, mocked_pipeline):
    client.app.dependency_overrides[get_optional_user_id] = lambda: "user-1"
    run_search, run_phase2 = mocked_pipeline

    res = client.post("/search", json={"query": "elon musk", "category": "Technology"})

    assert res.status_code == 200
    # Starlette's TestClient runs BackgroundTasks synchronously after the
    # response is built, so by the time we get here it's already fired.
    run_phase2.assert_awaited_once_with("s1")


def test_category_is_optional_when_the_client_omits_it_entirely(client, mocked_pipeline):
    # Regression: the frontend does `category: category || undefined`, which
    # JSON.stringify drops from the body entirely when no category is
    # selected. A required Body(...) here 422'd every such request in
    # production before ingestion.py's own "" == no-filter handling ever ran.
    run_search, _ = mocked_pipeline

    res = client.post("/search", json={"query": "elon musk"})

    assert res.status_code == 200
    assert run_search.call_args.kwargs["category"] == ""
