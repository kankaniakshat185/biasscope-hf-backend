"""app/services/extraction.py's extract_claims — the cached LLM call plus
JSON-parsing/repair around it. The LLM itself is mocked here (its
prompting behavior can't be unit tested); what's under test is that this
function handles the response shapes it actually has to deal with."""

from unittest.mock import AsyncMock

import pytest

from app.services import extraction as extraction_module


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(extraction_module, "cached_llm_call", mock)
    return mock


async def test_parses_a_well_formed_claims_response(mock_llm):
    mock_llm.return_value = '{"claims": [{"text": "Tesla filed for an IPO.", "claim_type": "EVENT", "confidence": 0.9, "evidence_sentence": "Tesla filed."}]}'
    claims = await extraction_module.extract_claims(prisma=None, article_text="...")
    assert len(claims) == 1
    assert claims[0]["claim_type"] == "EVENT"


async def test_accepts_bare_array_without_wrapper_object(mock_llm):
    mock_llm.return_value = '[{"text": "A claim.", "claim_type": "EVENT", "confidence": 0.8}]'
    claims = await extraction_module.extract_claims(prisma=None, article_text="...")
    assert len(claims) == 1


async def test_repairs_a_truncated_response_instead_of_returning_nothing(mock_llm):
    mock_llm.return_value = '{"claims": [{"text": "A."}, {"text": "B."}'  # cut off
    claims = await extraction_module.extract_claims(prisma=None, article_text="...")
    assert len(claims) >= 1
    assert claims[0]["text"] == "A."


async def test_unsalvageable_garbage_returns_empty_list_not_a_crash(mock_llm):
    mock_llm.return_value = "not json at all"
    claims = await extraction_module.extract_claims(prisma=None, article_text="...")
    assert claims == []


async def test_empty_llm_response_returns_empty_list(mock_llm):
    mock_llm.return_value = ""
    claims = await extraction_module.extract_claims(prisma=None, article_text="...")
    assert claims == []


async def test_passes_the_article_text_truncated_to_4000_chars(mock_llm):
    mock_llm.return_value = '{"claims": []}'
    long_text = "x" * 10000
    await extraction_module.extract_claims(prisma=None, article_text=long_text)
    user_prompt = mock_llm.call_args.args[3]
    assert len(user_prompt) < 4100  # 4000 chars of article + a short prefix
