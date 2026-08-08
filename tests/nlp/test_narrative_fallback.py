"""app/services/nlp.py's generate_narrative / generate_contrastive_summaries
— specifically their no-HF_TOKEN fallback paths, which must always
produce *something* deterministic rather than leaving the dashboard blank
when the LLM is unavailable."""

import pytest

from app.services.nlp import generate_contrastive_summaries, generate_narrative


def _article(bias_label, sentiment, source="example.com"):
    return {"title": f"{bias_label} headline", "source": source, "bias_label": bias_label, "sentiment": sentiment}


@pytest.fixture(autouse=True)
def no_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


async def test_narrative_falls_back_to_deterministic_summary_without_a_token():
    articles = [_article("LEFT", "negative"), _article("RIGHT", "positive"), _article("CENTER", "neutral")]
    summary = await generate_narrative(prisma=None, articles=articles)
    assert isinstance(summary, str)
    assert len(summary) > 0


async def test_narrative_with_no_articles_says_so_without_crashing():
    summary = await generate_narrative(prisma=None, articles=[])
    assert "No narrative available" in summary


async def test_narrative_fallback_never_calls_the_llm_client(monkeypatch):
    from app.services import nlp as nlp_module
    called = False

    async def fake_cached_llm_call(*args, **kwargs):
        nonlocal called
        called = True
        return "should not be reached"

    monkeypatch.setattr(nlp_module, "cached_llm_call", fake_cached_llm_call)
    await generate_narrative(prisma=None, articles=[_article("LEFT", "negative")])
    assert called is False


async def test_contrastive_summaries_without_a_token_says_so_for_both_sides():
    result = await generate_contrastive_summaries(prisma=None, articles=[_article("LEFT", "negative")])
    assert "left" in result and "right" in result
    assert "No token available" in result["left"]
    assert "No token available" in result["right"]
