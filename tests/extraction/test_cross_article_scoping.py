"""app/services/extraction.py's process_and_store_claims — specifically the
cross-article dedup match query's topic scoping (R4).

This used to search the ENTIRE `claim` table for a cosine-similar match
with no topic filter at all, while run_claim_clustering (clustering.py) was
already deliberately scoped by `search.query`. A claim from a completely
unrelated search could merge into this one on embedding similarity alone,
attaching that unrelated search's evidence to what this search's
intelligence report believes is its own claim. See AUDIT_TASKS.md R4.

Real embedding model calls (embed_text) and the LLM extraction call
(cached_llm_call) are mocked here — the interesting behavior under test is
which SQL this function sends to `prisma.query_raw`, not embedding quality
(that's covered for real in tests/extraction/test_dedup.py and
tests/clustering/test_similarity.py).
"""

from unittest.mock import AsyncMock

import pytest

from app.services import extraction as extraction_module
from tests.fakes import FakePrisma


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    # Query 1: the existing_match vector search -> no match, take the
    # "create new claim" path. Query 2: the INSERT ... RETURNING id.
    # Query 3: the existing_evidence dedup check -> none yet.
    prisma.query_raw = AsyncMock(side_effect=[[], [{"id": "new-claim-1"}], []])
    return prisma


@pytest.fixture(autouse=True)
def mocked_model_calls(monkeypatch):
    monkeypatch.setattr(extraction_module, "embed_text", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        extraction_module, "cached_llm_call",
        AsyncMock(return_value='{"claims": [{"text": "Tesla filed for an IPO worth $75 billion.", "claim_type": "EVENT", "confidence": 0.9, "evidence_sentence": "Tesla filed for an IPO."}]}'),
    )


async def test_dedup_match_is_scoped_to_the_current_query(fake_prisma):
    await extraction_module.process_and_store_claims(
        fake_prisma, "article-1", "Tesla filed for an IPO worth $75 billion on Monday, the company confirmed in a filing.",
        "reuters.com", "https://reuters.com/a", None, query="elon musk tesla",
    )

    existing_match_call = fake_prisma.query_raw.call_args_list[0]
    sql, vector_string, threshold, query_arg = existing_match_call.args
    assert "JOIN \"evidence\"" in sql
    assert "JOIN \"search\"" in sql
    # Production regression: `SELECT DISTINCT` combined with an ORDER BY
    # expression not in the SELECT list is a real Postgres syntax error
    # ("for SELECT DISTINCT, ORDER BY expressions must appear in select
    # list") — this shipped and broke every Phase 2 extraction in
    # production, invisible here because FakePrisma's query_raw never
    # actually executes SQL against a real database. This assertion can
    # only catch "someone re-added DISTINCT," not validate real Postgres
    # semantics — there is no substitute for smoke-testing new/changed
    # query_raw SQL against a real Postgres+pgvector instance before it
    # reaches production.
    assert "DISTINCT" not in sql
    assert "LOWER(s.query) = LOWER($3)" in sql
    assert threshold == extraction_module.CROSS_ARTICLE_DEDUP_THRESHOLD
    assert query_arg == "elon musk tesla"


async def test_dedup_match_falls_back_to_global_search_when_no_query_given(fake_prisma):
    # Matches run_claim_clustering's `query=None` escape hatch (used by the
    # /debug/rerun-* admin tools) — an empty/falsy query means "search
    # everything," not "search nothing."
    await extraction_module.process_and_store_claims(
        fake_prisma, "article-1", "Tesla filed for an IPO worth $75 billion on Monday, the company confirmed in a filing.",
        "reuters.com", "https://reuters.com/a", None, query="",
    )

    existing_match_call = fake_prisma.query_raw.call_args_list[0]
    sql = existing_match_call.args[0]
    assert "JOIN" not in sql
    # Only the vector + threshold params, no query param.
    assert len(existing_match_call.args) == 3


async def test_new_claim_is_created_and_evidence_attached(fake_prisma):
    inserted = await extraction_module.process_and_store_claims(
        fake_prisma, "article-1", "Tesla filed for an IPO worth $75 billion on Monday, the company confirmed in a filing.",
        "reuters.com", "https://reuters.com/a", None, query="tesla ipo",
    )

    assert inserted == ["new-claim-1"]
    fake_prisma.evidence.create.assert_awaited_once()
    evidence_data = fake_prisma.evidence.create.call_args.kwargs["data"]
    assert evidence_data["claimId"] == "new-claim-1"
    assert evidence_data["articleId"] == "article-1"
    assert evidence_data["source"] == "reuters.com"
