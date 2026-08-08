"""
Shared fixtures.

A few of these load real ML models (sentence-transformers, spacy, the NLI
cross-encoder) rather than mocking them — see AUDIT_TASKS.md T1 for why:
the backend README already claimed a test suite existed that validated
real embedding-similarity thresholds and real cross-encoder behavior, so
these tests earn that claim by actually doing so rather than asserting
against mocks that could drift from what the real models do. They're
session-scoped so the (one-time, few-second) model load cost is paid once
per test run, not once per test.
"""

import pytest


@pytest.fixture(scope="session")
def embedding_model():
    """The real sentence-transformers model extraction.py uses for
    within-article and cross-article claim dedup."""
    from app.services.extraction import get_embedding_model
    return get_embedding_model()


@pytest.fixture(scope="session")
def nli_classifier():
    """The real NLI cross-encoder clustering.py uses for contradiction
    detection — also what tests/nlp/test_grounding.py repurposes to check
    that a claim doesn't contradict its own supporting evidence sentence."""
    from app.services.clustering import get_nli_classifier
    clf = get_nli_classifier()
    if not clf:
        pytest.skip("NLI model failed to load (no network / model cache available)")
    return clf


def make_article(
    title="Sample Headline",
    content="Sample article content that is long enough to pass validation checks easily.",
    source="example.com",
    url="https://example.com/article",
    sentiment="neutral",
    sentiment_score=0.0,
    bias_label="UNKNOWN",
    **overrides,
):
    """Builds a plain-dict article in the shape the pipeline passes between
    stages (services/*.py operate on dicts, not Pydantic models, throughout)."""
    article = {
        "title": title,
        "content": content,
        "source": source,
        "url": url,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "bias_label": bias_label,
    }
    article.update(overrides)
    return article
