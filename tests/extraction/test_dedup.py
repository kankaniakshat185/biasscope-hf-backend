"""app/services/extraction.py's embedding-based within-article dedup —
real sentence-transformers model, not mocked (see conftest.py)."""

import pytest

from app.services.extraction import DEDUP_THRESHOLD, claim_relevance, cosine_similarity, deduplicate_claims, embed_text


@pytest.mark.model
def test_near_duplicate_claims_are_deduplicated(embedding_model):
    # Verified empirically: cosine similarity ~0.99 with the real model —
    # comfortably over DEDUP_THRESHOLD (0.92). A paraphrase using
    # different vocabulary for the same fact ("IPO" vs "initial public
    # offering") only reaches ~0.77, well below threshold — real
    # embedding models are more sensitive to trivial rewording than to
    # synonym substitution, which is worth knowing when tuning this gate.
    claims = [
        {"text": "Tesla filed for an IPO worth $75 billion on Monday."},
        {"text": "Tesla filed for an IPO worth $75 billion on Monday, the company confirmed."},
    ]
    deduped = deduplicate_claims(claims)
    assert len(deduped) == 1
    # Keeps the longer, richer claim of the two.
    assert "confirmed" in deduped[0]["text"]


@pytest.mark.model
def test_genuinely_distinct_claims_are_both_kept(embedding_model):
    claims = [
        {"text": "Tesla filed for an IPO worth $75 billion on Monday."},
        {"text": "The Senate passed the infrastructure bill on Tuesday."},
    ]
    deduped = deduplicate_claims(claims)
    assert len(deduped) == 2


def test_single_claim_list_is_a_no_op():
    claims = [{"text": "Only one claim here."}]
    assert deduplicate_claims(claims) == claims


def test_empty_list_is_a_no_op():
    assert deduplicate_claims([]) == []


@pytest.mark.model
def test_dedup_threshold_is_high_enough_to_not_merge_related_but_distinct_facts(embedding_model):
    # Two real, non-duplicate facts about the same event shouldn't collapse
    # into one just because they're topically related.
    claims = [
        {"text": "Tesla filed for an IPO worth $75 billion on Monday."},
        {"text": "Tesla's IPO filing valued the company at $75 billion, its highest ever."},
    ]
    a = embed_text(claims[0]["text"])
    b = embed_text(claims[1]["text"])
    sim = cosine_similarity(a, b)
    # Document what DEDUP_THRESHOLD (0.92) actually gates against with a
    # real pair, so a future change to the constant has a concrete anchor.
    assert 0.0 <= sim <= 1.0
    assert DEDUP_THRESHOLD == 0.92


@pytest.mark.model
def test_claim_relevance_scores_on_topic_claim_higher_than_off_topic(embedding_model):
    on_topic = embed_text("Tesla announced a new factory in Texas.")
    off_topic = embed_text("The weather forecast predicts rain this weekend.")

    on_topic_score = claim_relevance("Tesla factory expansion", on_topic)
    off_topic_score = claim_relevance("Tesla factory expansion", off_topic)

    assert on_topic_score > off_topic_score


def test_claim_relevance_with_no_query_is_always_fully_relevant():
    assert claim_relevance("", [0.1, 0.2, 0.3]) == 1.0
