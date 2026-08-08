"""app/services/intelligence.py — get_results() is where the A3 fix
(per-article reliability tier from the backend's registry, replacing the
frontend's now-deleted hardcoded/disagreeing domain arrays) lives, and
get_search_intelligence() is where the F4 fix (phase2Status exposed so
the frontend can stop polling) lives. Both get direct regression coverage
here rather than relying on the router/E2E layer to catch a regression.
"""

import pytest
from fastapi import HTTPException

from app.services import intelligence as intelligence_module
from tests.fakes import FakePrisma, fake_article, fake_claim, fake_cluster, fake_event, fake_evidence, fake_search


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(intelligence_module, "prisma", prisma)
    return prisma


# ── get_results ───────────────────────────────────────────────────────

async def test_get_results_404s_for_unknown_search(fake_prisma):
    fake_prisma.search.find_unique.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        await intelligence_module.get_results("nope")
    assert exc_info.value.status_code == 404


async def test_get_results_attaches_reliability_tier_per_article(fake_prisma):
    search = fake_search(articles=[
        fake_article(id="a1", source="reuters.com"),
        fake_article(id="a2", source="totally-unknown-blog.xyz"),
        fake_article(id="a3", source="breitbart.com"),
    ])
    fake_prisma.search.find_unique.return_value = search

    data = await intelligence_module.get_results("search-1")

    by_id = {a["id"]: a for a in data["articles"]}
    assert by_id["a1"]["reliabilityTier"] == "High"       # reuters.com, real registry entry
    assert by_id["a2"]["reliabilityTier"] == "Unknown"    # not in the registry
    assert by_id["a3"]["reliabilityTier"] == "Low"        # breitbart.com, real registry entry
    assert 0.0 <= by_id["a1"]["reliabilityScore"] <= 1.0


async def test_get_results_returns_a_plain_dict_not_a_model_instance(fake_prisma):
    # Regression: this used to return the raw Prisma model, which FastAPI
    # happened to serialize correctly but which create_demo_snapshot.py
    # then mishandled (see the Json bug in AUDIT_TASKS.md's Tooling
    # section). Explicitly a dict now.
    fake_prisma.search.find_unique.return_value = fake_search(articles=[fake_article()])
    data = await intelligence_module.get_results("search-1")
    assert isinstance(data, dict)
    assert isinstance(data["articles"], list)
    assert isinstance(data["articles"][0], dict)


# ── get_search_intelligence ──────────────────────────────────────────

async def test_get_search_intelligence_handles_no_articles_yet(fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search(phase2Status="processing")
    fake_prisma.article.find_many.return_value = []

    result = await intelligence_module.get_search_intelligence("search-1")

    assert result == {"events": [], "clusters": [], "claims": [], "metrics": {}, "status": "processing"}


async def test_get_search_intelligence_reports_status_for_the_polling_frontend(fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search(phase2Status="complete")
    fake_prisma.article.find_many.return_value = [fake_article()]
    fake_prisma.evidence.find_many.return_value = []

    result = await intelligence_module.get_search_intelligence("search-1")

    assert result["status"] == "complete"


async def test_get_search_intelligence_defaults_status_to_complete_for_demo_snapshots(fake_prisma):
    # No backing Search row (demo snapshots are static, pre-baked data) —
    # must not crash, and should read as "done" rather than "processing"
    # forever.
    fake_prisma.search.find_unique.return_value = None
    fake_prisma.article.find_many.return_value = []

    result = await intelligence_module.get_search_intelligence("demo-search-1")

    assert result["status"] == "complete"


async def test_get_search_intelligence_builds_the_claim_cluster_event_graph(fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search(phase2Status="complete")
    fake_prisma.article.find_many.return_value = [fake_article(id="a1")]

    ev1 = fake_evidence(id="ev1", claimId="claim-1", articleId="a1", source="reuters.com", sentence="Reuters says X.")
    ev2 = fake_evidence(id="ev2", claimId="claim-1", articleId="a1", source="apnews.com", sentence="AP says X too.")
    fake_prisma.evidence.find_many.return_value = [ev1, ev2]

    cluster = fake_cluster(id="cluster-1", canonicalClaim="X happened.", consensusScore=0.9)
    event = fake_event(id="event-1", title="X Event", importanceScore=0.8)
    cluster.eventId = "event-1"
    cluster.event = event
    claim = fake_claim(id="claim-1", clusterId="cluster-1", cluster=cluster, evidence=[ev1, ev2])
    fake_prisma.claim.find_many.return_value = [claim]

    result = await intelligence_module.get_search_intelligence("search-1")

    assert result["metrics"]["articlesProcessed"] == 1
    assert result["metrics"]["canonicalClaims"] == 1
    assert result["metrics"]["clusters"] == 1
    assert result["metrics"]["events"] == 1

    assert len(result["claims"]) == 1
    assert result["claims"][0]["evidenceCount"] == 2
    assert set(result["claims"][0]["sources"]) == {"reuters.com", "apnews.com"}

    assert result["clusters"][0]["sourceCount"] == 2
    assert result["events"][0]["title"] == "X Event"
    assert result["events"][0]["sourceCount"] == 2


async def test_get_search_intelligence_deduplicates_evidence_by_sentence_prefix(fake_prisma):
    fake_prisma.search.find_unique.return_value = fake_search()
    fake_prisma.article.find_many.return_value = [fake_article(id="a1")]

    # Same sentence text reaching the cluster via two different claims —
    # should collapse to one entry in the cluster's evidence list.
    ev = fake_evidence(id="ev1", claimId="claim-1", articleId="a1", sentence="Identical sentence text here.")
    fake_prisma.evidence.find_many.return_value = [ev]

    cluster = fake_cluster(id="cluster-1")
    claim = fake_claim(id="claim-1", clusterId="cluster-1", cluster=cluster, evidence=[ev, ev])
    fake_prisma.claim.find_many.return_value = [claim]

    result = await intelligence_module.get_search_intelligence("search-1")

    assert result["clusters"][0]["evidenceCount"] == 1
