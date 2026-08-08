"""app/routers/history.py — always scoped to the authenticated caller.
DELETE's ownership check (added as part of S2; it didn't exist before)
gets the most scrutiny here since it was a real, previously-missing IDOR."""

import pytest

from app.deps.auth import get_current_user_id
from app.routers import history as history_module
from tests.fakes import FakePrisma, fake_search
from tests.routers.conftest import make_client


@pytest.fixture
def client():
    return make_client(history_module.router)


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(history_module, "prisma", prisma)
    return prisma


def _authed_as(app, user_id):
    app.dependency_overrides[get_current_user_id] = lambda: user_id


# ── GET /history ──────────────────────────────────────────────────────

def test_get_history_401s_with_no_session(client, fake_prisma, monkeypatch):
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())  # no session found
    res = client.get("/history")
    assert res.status_code == 401


def test_get_history_returns_only_the_caller_own_searches(client, fake_prisma):
    fake_prisma.search.count.return_value = 2
    fake_prisma.search.find_many.return_value = [fake_search(id="s1"), fake_search(id="s2")]
    _authed_as(client.app, "user-1")

    res = client.get("/history")

    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert [s["id"] for s in body["searches"]] == ["s1", "s2"]
    # The actual filter passed to Prisma must scope by the AUTHENTICATED
    # user, not anything the client could have supplied — this is the
    # entire point of the S2 fix.
    where_clause = fake_prisma.search.find_many.call_args.kwargs["where"]
    assert where_clause == {"userId": "user-1"}


def test_get_history_respects_pagination_params(client, fake_prisma):
    fake_prisma.search.count.return_value = 100
    fake_prisma.search.find_many.return_value = []
    _authed_as(client.app, "user-1")

    res = client.get("/history?limit=10&offset=20")

    assert res.status_code == 200
    body = res.json()
    assert body["limit"] == 10
    assert body["offset"] == 20
    kwargs = fake_prisma.search.find_many.call_args.kwargs
    assert kwargs["take"] == 10
    assert kwargs["skip"] == 20


def test_get_history_rejects_limit_above_the_cap(client, fake_prisma):
    _authed_as(client.app, "user-1")
    res = client.get("/history?limit=99999")
    assert res.status_code == 422  # FastAPI's own Query(le=...) validation


# ── DELETE /history/{search_id} ──────────────────────────────────────

def test_delete_401s_with_no_session(client, fake_prisma, monkeypatch):
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())
    res = client.delete("/history/some-id")
    assert res.status_code == 401


def test_delete_404s_for_nonexistent_search(client, fake_prisma):
    fake_prisma.search.find_unique.return_value = None
    _authed_as(client.app, "user-1")

    res = client.delete("/history/does-not-exist")

    assert res.status_code == 404
    fake_prisma.search.delete.assert_not_called()


def test_delete_403s_when_caller_does_not_own_the_search(client, fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", userId="someone-else")
    _authed_as(client.app, "user-1")

    res = client.delete("/history/s1")

    assert res.status_code == 403
    fake_prisma.search.delete.assert_not_called()


def test_delete_succeeds_for_the_actual_owner(client, fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", userId="user-1")
    _authed_as(client.app, "user-1")

    res = client.delete("/history/s1")

    assert res.status_code == 200
    fake_prisma.search.delete.assert_awaited_once_with(where={"id": "s1"})
