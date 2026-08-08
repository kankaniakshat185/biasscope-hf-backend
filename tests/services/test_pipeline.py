"""app/services/pipeline.py — the /search orchestration (extracted from
main.py in the A1 refactor) and the Phase 2 background pipeline, whose
phase2Status transitions are the F4 fix that lets the frontend stop
polling once the pipeline is actually done."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.services import pipeline as pipeline_module
from tests.fakes import FakePrisma, fake_search


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(pipeline_module, "prisma", prisma)
    return prisma


# ── build_article_create_payload ───────────────────────────────────────
# R1/R3: this used to be inlined separately in both run_search_pipeline and
# app/tasks/snapshot_task.py — the snapshot copy read the wrong (camelCase)
# keys off the analyzed-article dict, so every value it wrote came back
# NULL. Extracted to one function both call; these tests pin the exact
# snake_case → camelCase mapping analyze_articles()'s real output requires.

def test_build_article_create_payload_maps_analyzed_snake_case_fields_to_prisma_columns():
    art = {
        "title": "Tesla Files for IPO",
        "content": "Tesla filed for an IPO worth $75 billion.",
        "source": "reuters.com",
        "url": "https://reuters.com/a",
        "sentiment": "positive",
        "sentiment_score": 0.42,
        "bias_label": "CENTER",
        "source_bias": "CENTER",
        "deviation_score": 0.0,
        "entities": {"Tesla": "ORG"},
        "published_at": "2024-01-01T00:00:00Z",
    }

    payload = pipeline_module.build_article_create_payload("search-1", art)

    assert payload["searchId"] == "search-1"
    assert payload["title"] == "Tesla Files for IPO"
    assert payload["sentimentScore"] == 0.42
    assert payload["biasLabel"] == "CENTER"
    assert payload["sourceBias"] == "CENTER"
    assert payload["publishedAt"] == "2024-01-01T00:00:00Z"


def test_build_article_create_payload_defaults_missing_fields_instead_of_raising():
    # A near-empty analyzed dict (e.g. scraping failed and analyze_articles
    # short-circuited) must still produce a valid create payload, not a
    # KeyError — this used to be `art["title"]` (raises) in the old
    # snapshot_task.py implementation.
    payload = pipeline_module.build_article_create_payload("search-1", {})

    assert payload["title"] == "No Title"
    assert payload["source"] == "Unknown"
    assert payload["biasLabel"] == "UNKNOWN"
    assert payload["sourceBias"] == "UNKNOWN"
    assert payload["sentimentScore"] == 0.0
    assert payload["publishedAt"] is None


@pytest.fixture
def mocked_stages(monkeypatch):
    """Everything run_search_pipeline calls besides the DB writes
    themselves — network (ingestion) and LLM (narrative) calls."""
    monkeypatch.setattr(pipeline_module, "ingest_articles", AsyncMock(return_value=[
        {"title": "Tesla Files for IPO", "url": "https://reuters.com/a", "source": "reuters.com",
         "content": "Tesla filed for an IPO worth $75 billion on Monday.", "published_at": None},
    ]))
    monkeypatch.setattr(pipeline_module, "generate_narrative", AsyncMock(return_value="A narrative summary."))
    monkeypatch.setattr(pipeline_module, "generate_contrastive_summaries",
                         AsyncMock(return_value={"left": "left take", "right": "right take"}))


# ── run_search_pipeline ───────────────────────────────────────────────

async def test_run_search_pipeline_404s_when_ingestion_finds_nothing(fake_prisma, monkeypatch):
    monkeypatch.setattr(pipeline_module, "ingest_articles", AsyncMock(return_value=[]))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_module.run_search_pipeline(
            query="elon musk", category="Technology", domains=None,
            exclude_domains=None, from_date=None, to_date=None, user_id=None,
        )
    assert exc_info.value.status_code == 404


async def test_run_search_pipeline_persists_search_with_the_given_user(fake_prisma, mocked_stages):
    fake_prisma.search.create.return_value = fake_search(id="new-search-id", userId="user-1")

    result = await pipeline_module.run_search_pipeline(
        query="elon musk", category="Technology", domains=None,
        exclude_domains=None, from_date=None, to_date=None, user_id="user-1",
    )

    assert result["search_id"] == "new-search-id"
    create_kwargs = fake_prisma.search.create.call_args.kwargs
    assert create_kwargs["data"]["userId"] == "user-1"
    assert create_kwargs["data"]["query"] == "elon musk"
    fake_prisma.article.create_many.assert_awaited_once()
    fake_prisma.insight.create.assert_awaited_once()


async def test_run_search_pipeline_allows_anonymous_user(fake_prisma, mocked_stages):
    fake_prisma.search.create.return_value = fake_search(id="new-search-id", userId=None)

    await pipeline_module.run_search_pipeline(
        query="elon musk", category="Technology", domains=None,
        exclude_domains=None, from_date=None, to_date=None, user_id=None,
    )

    assert fake_prisma.search.create.call_args.kwargs["data"]["userId"] is None


# ── run_phase2_pipeline: phase2Status transitions (the F4 fix) ────────

@pytest.fixture
def mocked_phase2_stages(monkeypatch):
    monkeypatch.setattr(pipeline_module, "process_and_store_claims", AsyncMock())
    monkeypatch.setattr(pipeline_module, "run_claim_clustering", AsyncMock())
    monkeypatch.setattr(pipeline_module, "run_event_detection", AsyncMock())


async def test_phase2_pipeline_sets_processing_then_complete_on_success(fake_prisma, mocked_phase2_stages, monkeypatch):
    monkeypatch.setenv("SKIP_EXTRACTION", "1")  # skip the article-fetch loop, not under test here
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", query="elon musk")

    await pipeline_module.run_phase2_pipeline("s1")

    statuses = [
        call.kwargs["data"]["phase2Status"]
        for call in fake_prisma.search.update.call_args_list
    ]
    assert statuses == ["processing", "complete"]


async def test_phase2_pipeline_sets_failed_on_exception(fake_prisma, monkeypatch):
    monkeypatch.setenv("SKIP_EXTRACTION", "1")
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", query="elon musk")
    monkeypatch.setattr(
        pipeline_module, "run_claim_clustering",
        AsyncMock(side_effect=RuntimeError("clustering blew up")),
    )

    # Errors are caught and logged inside run_phase2_pipeline — it must
    # not raise (it runs as an untracked FastAPI BackgroundTask; an
    # unhandled exception there is silently swallowed by the framework
    # with no status ever recorded, which is worse).
    await pipeline_module.run_phase2_pipeline("s1")

    statuses = [
        call.kwargs["data"]["phase2Status"]
        for call in fake_prisma.search.update.call_args_list
    ]
    assert statuses == ["processing", "failed"]


async def test_phase2_pipeline_respects_skip_flags(fake_prisma, mocked_phase2_stages, monkeypatch):
    monkeypatch.setenv("SKIP_EXTRACTION", "1")
    monkeypatch.setenv("SKIP_CLUSTERING", "1")
    monkeypatch.setenv("SKIP_EVENTS", "1")
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", query="elon musk")

    await pipeline_module.run_phase2_pipeline("s1")

    pipeline_module.process_and_store_claims.assert_not_called()
    pipeline_module.run_claim_clustering.assert_not_called()
    pipeline_module.run_event_detection.assert_not_called()
    # Still must land on "complete" — skipping stages isn't a failure.
    statuses = [call.kwargs["data"]["phase2Status"] for call in fake_prisma.search.update.call_args_list]
    assert statuses[-1] == "complete"


async def test_phase2_pipeline_scopes_clustering_to_the_search_query(fake_prisma, mocked_phase2_stages, monkeypatch):
    monkeypatch.setenv("SKIP_EXTRACTION", "1")
    fake_prisma.search.find_unique.return_value = fake_search(id="s1", query="ai regulation")

    await pipeline_module.run_phase2_pipeline("s1")

    # P1 fix: clustering must be scoped to this search's query, not global.
    pipeline_module.run_claim_clustering.assert_awaited_once_with(fake_prisma, "ai regulation")
