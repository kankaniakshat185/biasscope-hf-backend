"""app/routers/chat.py — the two chat endpoints unified into one
_chat_with_context helper (Q3/A4 fix: they used to be near-duplicated code
each bypassing the LLM cache)."""

from unittest.mock import AsyncMock

import pytest

from app.routers import chat as chat_module
from tests.fakes import FakePrisma, fake_article
from tests.routers.conftest import make_client


@pytest.fixture
def client():
    return make_client(chat_module.router)


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(chat_module, "prisma", prisma)
    return prisma


@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(return_value="A helpful answer.")
    monkeypatch.setattr(chat_module, "cached_llm_call", mock)
    return mock


def test_chat_with_article_404s_for_unknown_article(client, fake_prisma):
    fake_prisma.article.find_unique.return_value = None
    res = client.post("/chat-with-article", json={"articleId": "nope", "message": "What happened?"})
    assert res.status_code == 404


def test_chat_with_article_uses_article_content_as_context(client, fake_prisma, mock_llm):
    fake_prisma.article.find_unique.return_value = fake_article(content="Tesla filed for an IPO.")

    res = client.post("/chat-with-article", json={"articleId": "a1", "message": "What did they file for?"})

    assert res.status_code == 200
    assert res.json()["answer"] == "A helpful answer."
    system_prompt = mock_llm.call_args.args[2]
    assert "Tesla filed for an IPO" in system_prompt
    # Routed through the shared cached client — this used to construct its
    # own uncached InferenceClient, invisible to /debug/llm-usage.
    assert mock_llm.call_args.args[1] == "chat_article"


def test_chat_with_summary_404s_when_no_insight_exists(client, fake_prisma):
    fake_prisma.insight.find_first.return_value = None
    res = client.post("/chat-with-summary", json={"searchId": "s1", "message": "Summarize this."})
    assert res.status_code == 404


def test_chat_returns_a_helpful_message_when_the_llm_is_unavailable(client, fake_prisma, monkeypatch):
    fake_prisma.article.find_unique.return_value = fake_article()
    monkeypatch.setattr(chat_module, "cached_llm_call", AsyncMock(return_value=""))

    res = client.post("/chat-with-article", json={"articleId": "a1", "message": "hi"})

    assert res.status_code == 200
    assert "HF_TOKEN" in res.json()["answer"] or "reach" in res.json()["answer"].lower()
