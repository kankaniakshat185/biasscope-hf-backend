"""
Clustering/dedup similarity thresholds — the test the backend README has
claimed existed since before this suite was written ("Validates the
cosine similarity threshold ... to ensure distinct claims are not
improperly merged into the same canonical claim").

The README's own description said "0.85" — that number doesn't match any
threshold actually in the code. The three real ones, and what each one
gates:
  - extraction.DEDUP_THRESHOLD (0.92)              — same-article near-dup claims
  - extraction.CROSS_ARTICLE_DEDUP_THRESHOLD (0.88) — merge into an existing DB claim
  - clustering.COHESION_THRESHOLD (0.72)            — cluster is a real "event", not just a topic

Uses the real sentence-transformers model (see conftest.py's
embedding_model fixture) — these thresholds are meaningless against
anything else.
"""

import numpy as np
import pytest

from app.services.clustering import COHESION_THRESHOLD, compute_cluster_cohesion
from app.services.extraction import CROSS_ARTICLE_DEDUP_THRESHOLD, DEDUP_THRESHOLD, embed_text


@pytest.mark.model
def test_thresholds_have_the_expected_values():
    # Pins the actual values so a silent edit to any of them is visible in
    # a diff of a test file, not just in schema.prisma comments.
    assert DEDUP_THRESHOLD == 0.92
    assert CROSS_ARTICLE_DEDUP_THRESHOLD == 0.88
    assert COHESION_THRESHOLD == 0.72


@pytest.mark.model
def test_cross_article_threshold_is_looser_than_within_article_threshold():
    # By design: within-article dedup (same LLM call, same article) can
    # demand near-exact rewording; cross-article dedup (different
    # articles, different journalists) has to tolerate more variation to
    # actually catch the same underlying fact reported twice.
    assert CROSS_ARTICLE_DEDUP_THRESHOLD < DEDUP_THRESHOLD


@pytest.mark.model
def test_cohesion_high_for_tightly_related_claims_about_one_event():
    vectors = np.array([
        embed_text("Tesla filed for an IPO worth $75 billion on Monday."),
        embed_text("Tesla has filed for an IPO worth $75 billion on Monday."),
        embed_text("Tesla filed for an IPO worth $75 billion on Monday, the company confirmed."),
    ])
    cohesion = compute_cluster_cohesion(vectors)
    assert cohesion >= COHESION_THRESHOLD


@pytest.mark.model
def test_cohesion_low_for_claims_that_only_share_a_topic_not_an_event():
    # Same entity (Tesla), three different, unrelated stories about it —
    # exactly the "loose topic grouping" the cohesion gate exists to reject.
    vectors = np.array([
        embed_text("Tesla filed for an IPO worth $75 billion on Monday."),
        embed_text("Tesla recalled 50,000 vehicles over a braking defect."),
        embed_text("Tesla's stock price fell 8% after the earnings call."),
    ])
    cohesion = compute_cluster_cohesion(vectors)
    assert cohesion < COHESION_THRESHOLD


def test_cohesion_of_a_single_vector_is_perfect_by_definition():
    vectors = np.array([[0.1, 0.2, 0.3]])
    assert compute_cluster_cohesion(vectors) == 1.0


def test_cohesion_of_identical_vectors_is_one():
    v = np.array([0.5, 0.5, 0.5])
    vectors = np.array([v, v, v])
    assert compute_cluster_cohesion(vectors) == pytest.approx(1.0, abs=1e-6)


def test_cohesion_of_orthogonal_vectors_is_zero():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert compute_cluster_cohesion(vectors) == pytest.approx(0.0, abs=1e-6)
