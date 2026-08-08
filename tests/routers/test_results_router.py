"""app/routers/results.py — thin route wrappers around
app/services/intelligence.py; the interesting logic is tested there
directly. These confirm the routes are wired correctly and translate a
missing demo snapshot into a 404."""

import pytest

from app.routers import results as results_module
from tests.fakes import FakePrisma, FakeRecord
from tests.routers.conftest import make_client


@pytest.fixture
def client():
    return make_client(results_module.router)


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(results_module, "prisma", prisma)
    return prisma


def test_demo_snapshot_404s_for_unknown_topic(client, fake_prisma):
    fake_prisma.demosnapshot.find_unique.return_value = None
    res = client.get("/demo/nonexistent-topic")
    assert res.status_code == 404


def test_demo_snapshot_returns_the_stored_data_and_lowercases_the_topic(client, fake_prisma):
    fake_prisma.demosnapshot.find_unique.return_value = FakeRecord(topic="elon musk", data={"search": {}, "intelligence": {}})

    res = client.get("/demo/Elon%20Musk")

    assert res.status_code == 200
    where_clause = fake_prisma.demosnapshot.find_unique.call_args.kwargs["where"]
    assert where_clause == {"topic": "elon musk"}


def test_results_route_delegates_to_get_results(client, monkeypatch):
    from unittest.mock import AsyncMock
    mock = AsyncMock(return_value={"id": "s1", "articles": []})
    monkeypatch.setattr(results_module, "get_results", mock)

    res = client.get("/results/s1")

    assert res.status_code == 200
    assert res.json()["id"] == "s1"
    mock.assert_awaited_once_with("s1")


def test_intelligence_route_delegates_to_get_search_intelligence(client, monkeypatch):
    from unittest.mock import AsyncMock
    mock = AsyncMock(return_value={"events": [], "clusters": [], "claims": [], "metrics": {}, "status": "complete"})
    monkeypatch.setattr(results_module, "get_search_intelligence", mock)

    res = client.get("/results/s1/intelligence")

    assert res.status_code == 200
    assert res.json()["status"] == "complete"
    mock.assert_awaited_once_with("s1")
