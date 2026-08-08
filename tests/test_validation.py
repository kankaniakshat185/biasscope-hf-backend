"""app/services/validation.py — the Data Quality Score and polarization
(Jensen-Shannon Divergence) formulas the dashboard's own Methodology panel
documents to users. These tests use the exact worked examples from that
panel (dashboard/[id]/page.tsx's MethodologyReport component) as fixtures,
so a change here that breaks the documented math breaks a test, not just
the promise made to users.

Calling validate_articles() imports app.services.nlp for extract_keywords,
which loads real ML models at import time — first test to touch this
pays that one-time cost for the session.
"""

import pytest

from app.services.validation import validate_articles


def _article(source, content_len, bias_label="CENTER", sentiment_score=0.0):
    return {
        "source": source,
        "content": "x" * content_len,
        "bias_label": bias_label,
        "sentiment_score": sentiment_score,
        "entities": {},
        "title": f"Article from {source}",
    }


@pytest.mark.model
def test_data_quality_score_matches_documented_worked_example():
    # 10 articles, 5 unique sources, none missing content, avg length 1200.
    # Methodology panel: C=1.0, D=0.5, R=1.0 -> DQS = 0.4 + 0.15 + 0.3 = 0.85
    sources = [f"source{i}.com" for i in range(5)]
    articles = [_article(sources[i % 5], content_len=1200) for i in range(10)]

    result = validate_articles(articles)

    assert result["missing_content"] == 0
    assert result["data_quality_score"] == pytest.approx(0.85, abs=0.005)


@pytest.mark.model
def test_missing_content_penalizes_completeness():
    articles = [_article("source.com", content_len=1200) for _ in range(8)]
    articles += [_article("source.com", content_len=10) for _ in range(2)]  # < 50 chars = missing

    result = validate_articles(articles)

    assert result["missing_content"] == 2
    assert result["valid_articles"] == 8


@pytest.mark.model
def test_polarization_is_high_when_left_and_right_sentiment_diverge_completely():
    # Methodology panel's own example: Left = 100% negative, Right = 100%
    # positive -> JSD reaches its max (~ln(2)) -> polarization ~= 1.0.
    articles = [
        _article("left1.com", 1200, bias_label="LEFT", sentiment_score=-0.9),
        _article("left2.com", 1200, bias_label="LEFT", sentiment_score=-0.8),
        _article("right1.com", 1200, bias_label="RIGHT", sentiment_score=0.9),
        _article("right2.com", 1200, bias_label="RIGHT", sentiment_score=0.8),
    ]

    result = validate_articles(articles)

    assert result["polarization_score"] > 0.9


@pytest.mark.model
def test_polarization_is_low_when_left_and_right_sentiment_agree():
    articles = [
        _article("left1.com", 1200, bias_label="LEFT", sentiment_score=0.1),
        _article("left2.com", 1200, bias_label="LEFT", sentiment_score=0.15),
        _article("right1.com", 1200, bias_label="RIGHT", sentiment_score=0.1),
        _article("right2.com", 1200, bias_label="RIGHT", sentiment_score=0.12),
    ]

    result = validate_articles(articles)

    assert result["polarization_score"] < 0.3


@pytest.mark.model
def test_diversity_label_reflects_source_and_ideology_spread():
    # Methodology panel's own example: 6 articles, 3 US publishers (Left) +
    # 2 UK publishers (Center) = 5 unique sources, 2 countries, max
    # ideology 3/5 = 60% -> "High Diversity".
    articles = (
        [_article("nytimes.com", 1200, bias_label="LEFT"),
         _article("washingtonpost.com", 1200, bias_label="LEFT"),
         _article("cnn.com", 1200, bias_label="LEFT")]
        + [_article("bbc.co.uk", 1200, bias_label="CENTER"),
           _article("theguardian.com", 1200, bias_label="CENTER")]
    )

    result = validate_articles(articles)

    assert result["dataset_metrics"]["source_diversity"] == 5
    assert result["dataset_metrics"]["diversity_quality_label"] == "High Diversity"
