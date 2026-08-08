"""app/services/nlp.py's analyze_articles — R12 regression.

analyze_articles() mutates a dozen keys (sentiment, entities, source_bias,
bias_label, deviation_score, ...) on each article dict. It used to mutate
the caller's own dict objects in place, so the list passed in ended up
silently carrying analyzed-article fields too — harmless today only
because pipeline.py never re-reads its `cleaned_articles` list after
calling this, but a hidden coupling waiting for a future change to trip
over. Uses the real sentiment/bias/NER pipelines (loaded at module import
time) — mocking them away would defeat the point of an immutability check
on the function that actually does the mutating.
"""

import pytest

from app.services.nlp import analyze_articles


@pytest.mark.model
def test_does_not_mutate_the_caller_input_dicts():
    original = {"title": "Tesla Files for IPO", "content": "Tesla filed for an IPO worth $75 billion.", "source": "reuters.com"}
    articles = [original]

    analyzed = analyze_articles(articles)

    assert analyzed[0] is not original
    assert "sentiment" not in original
    assert "bias_label" not in original
    assert "entities" not in original
