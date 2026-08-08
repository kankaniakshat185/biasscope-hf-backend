"""app/routers/debug.py — this router is the literal fix for the audit's
#1 critical finding (unauthenticated destructive debug endpoints). These
tests exist specifically to catch a regression that reopens it: gating
must come before anything else runs, and must default to OFF.
"""

import pytest

from app.deps.auth import get_current_user_id
from app.routers import debug as debug_module
from tests.fakes import FakePrisma, fake_claim, fake_cluster, fake_event, fake_evidence
from tests.routers.conftest import make_client


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ROUTES", raising=False)
    return make_client(debug_module.router)


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(debug_module, "prisma", prisma)
    return prisma


def test_404s_when_disabled_even_with_no_auth_attempted(client, fake_prisma):
    # The default state: no env var set at all.
    res = client.get("/debug/status")
    assert res.status_code == 404


def test_404s_for_every_route_when_disabled(client, fake_prisma):
    for method, path in [
        ("get", "/debug/clusters"), ("get", "/debug/events"),
        ("get", "/debug/llm-usage"), ("get", "/debug/cache-stats"),
        ("post", "/debug/rerun-clustering"), ("post", "/debug/rerun-events"),
        ("post", "/debug/clear-cache"), ("post", "/debug/reset-phase2"),
        ("post", "/debug/rerun-full"), ("get", "/debug/run-one"),
        ("get", "/debug/cluster-quality"),
    ]:
        res = getattr(client, method)(path)
        assert res.status_code == 404, f"{method.upper()} {path} should 404 when ENABLE_DEBUG_ROUTES is unset"


def test_401s_when_enabled_but_not_logged_in(client, fake_prisma, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", "1")
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())  # no session
    res = client.get("/debug/status")
    assert res.status_code == 401


def test_reset_phase2_requires_both_flag_and_auth(client, fake_prisma):
    # Flag off, no auth override either — should 404 (flag checked first),
    # not accidentally wipe anything.
    res = client.post("/debug/reset-phase2")
    assert res.status_code == 404
    fake_prisma.query_raw.assert_not_called()


def test_works_end_to_end_when_flag_on_and_authenticated(client, fake_prisma, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", "1")
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    fake_prisma.query_raw.return_value = [{"cnt": 3}]

    res = client.get("/debug/status")

    assert res.status_code == 200
    assert res.json()["claims"] == 3


def test_cluster_quality_flags_low_evidence_single_source_clusters_as_noisy(client, fake_prisma, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", "1")
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-1"

    noisy = fake_cluster(
        id="c-noisy", consensusScore=0.1,
        claims=[fake_claim(id="claim-1", evidence=[fake_evidence(source="only-source.com")])],
    )
    fake_prisma.claimcluster.find_many.return_value = [noisy]

    res = client.get("/debug/cluster-quality")

    assert res.status_code == 200
    body = res.json()
    assert body["clusters"][0]["source_count"] == 1
    assert body["clusters"][0]["event_eligible"] is False
    assert body["clusters"][0]["noise_score"] >= 0.5
    assert body["noisy_clusters"] == 1


def test_events_endpoint_aggregates_sources_across_clusters(client, fake_prisma, monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", "1")
    client.app.dependency_overrides[get_current_user_id] = lambda: "user-1"

    cluster = fake_cluster(claims=[
        fake_claim(id="c1", evidence=[fake_evidence(source="reuters.com")]),
        fake_claim(id="c2", evidence=[fake_evidence(source="apnews.com")]),
    ])
    event = fake_event(claimClusters=[cluster])
    fake_prisma.event.find_many.return_value = [event]

    res = client.get("/debug/events")

    assert res.status_code == 200
    ev = res.json()["events"][0]
    assert ev["source_count"] == 2
    assert set(ev["sources"]) == {"reuters.com", "apnews.com"}
