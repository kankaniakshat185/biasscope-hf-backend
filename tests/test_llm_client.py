"""app/services/llm_client.py — the SHA-256 prompt cache and per-stage
usage analytics that the A4 fix made every LLM call in the app actually
go through (previously narrative/chat call sites bypassed it entirely,
so /debug/llm-usage silently under-reported real spend)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import llm_client
from tests.fakes import FakePrisma, FakeRecord


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(llm_client, "prisma", prisma, raising=False)
    return prisma


@pytest.fixture(autouse=True)
def reset_client_cache(monkeypatch):
    # _clients is a module-level dict caching InferenceClient instances by
    # model — clear it so one test's mock doesn't leak into another's.
    monkeypatch.setattr(llm_client, "_clients", {})


def test_clean_llm_response_strips_json_code_fence():
    assert llm_client._clean_llm_response('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_narrative_model_id_points_at_a_provider_that_actually_serves_it():
    # Regression: NARRATIVE_MODEL_ID used to be a distinct Llama-3 model id
    # that HF's Inference Router stopped routing to any enabled provider,
    # so every narrative/chat call failed with a 400 and silently fell back
    # in production. It's pinned to MODEL_ID now (confirmed working, since
    # the extraction stage calls it successfully) rather than a second,
    # independently-rotting model id.
    assert llm_client.NARRATIVE_MODEL_ID == llm_client.MODEL_ID


def test_clean_llm_response_strips_bare_code_fence():
    assert llm_client._clean_llm_response('```\nplain text\n```') == 'plain text'


def test_clean_llm_response_passes_through_unfenced_text():
    assert llm_client._clean_llm_response('already plain') == 'already plain'


def test_prompt_hash_is_deterministic():
    h1 = llm_client._compute_prompt_hash("model-a", "system", "user")
    h2 = llm_client._compute_prompt_hash("model-a", "system", "user")
    assert h1 == h2


def test_prompt_hash_differs_per_model_even_with_identical_prompts():
    # This is the whole point of including model in the hash — the same
    # prompt against two different models must not collide in the cache.
    h1 = llm_client._compute_prompt_hash("model-a", "system", "user")
    h2 = llm_client._compute_prompt_hash("model-b", "system", "user")
    assert h1 != h2


async def test_cache_hit_returns_cached_response_without_calling_the_model(fake_prisma, monkeypatch):
    fake_prisma.llmcache.find_unique.return_value = FakeRecord(response="cached answer")
    client_factory = MagicMock()
    monkeypatch.setattr(llm_client, "InferenceClient", client_factory)

    result = await llm_client.cached_llm_call(fake_prisma, "extraction", "sys", "user")

    assert result == "cached answer"
    client_factory.assert_not_called()
    usage_call = fake_prisma.llmusage.create.call_args.kwargs["data"]
    assert usage_call["cached"] is True
    assert usage_call["stage"] == "extraction"


async def test_cache_miss_with_no_hf_token_returns_empty_string(fake_prisma, monkeypatch):
    fake_prisma.llmcache.find_unique.return_value = None
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = await llm_client.cached_llm_call(fake_prisma, "extraction", "sys", "user")

    assert result == ""
    fake_prisma.llmcache.create.assert_not_called()


async def test_cache_miss_calls_the_model_and_stores_the_result(fake_prisma, monkeypatch):
    fake_prisma.llmcache.find_unique.return_value = None
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="```json\n{\"claims\": []}\n```"))]
    fake_client_instance = MagicMock(chat_completion=MagicMock(return_value=fake_response))
    monkeypatch.setattr(llm_client, "InferenceClient", MagicMock(return_value=fake_client_instance))

    result = await llm_client.cached_llm_call(fake_prisma, "extraction", "sys", "user", model="some-model")

    assert result == '{"claims": []}'
    create_kwargs = fake_prisma.llmcache.create.call_args.kwargs["data"]
    assert create_kwargs["response"] == '{"claims": []}'
    assert create_kwargs["model"] == "some-model"
    usage_kwargs = fake_prisma.llmusage.create.call_args.kwargs["data"]
    assert usage_kwargs["cached"] is False


async def test_uses_the_model_specific_client_not_a_shared_default(fake_prisma, monkeypatch):
    # Regression guard for the A4 fix: before, one InferenceClient was
    # shared for every stage regardless of which model was requested.
    fake_prisma.llmcache.find_unique.return_value = None
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    client_factory = MagicMock(return_value=MagicMock(chat_completion=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(llm_client, "InferenceClient", client_factory)

    await llm_client.cached_llm_call(fake_prisma, "narrative", "sys", "user", model=llm_client.NARRATIVE_MODEL_ID)
    await llm_client.cached_llm_call(fake_prisma, "extraction", "sys2", "user2", model=llm_client.MODEL_ID)

    called_models = [call.kwargs.get("model", call.args[0] if call.args else None) for call in client_factory.call_args_list]
    assert llm_client.NARRATIVE_MODEL_ID in called_models
    assert llm_client.MODEL_ID in called_models


async def test_cache_lookup_failure_falls_through_to_a_real_call_instead_of_crashing(fake_prisma, monkeypatch):
    fake_prisma.llmcache.find_unique = AsyncMock(side_effect=RuntimeError("db unavailable"))
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    monkeypatch.setattr(llm_client, "InferenceClient", MagicMock(return_value=MagicMock(chat_completion=MagicMock(return_value=fake_response))))

    result = await llm_client.cached_llm_call(fake_prisma, "extraction", "sys", "user")

    assert result == "ok"
