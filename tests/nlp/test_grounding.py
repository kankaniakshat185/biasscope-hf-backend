"""
"Hallucination Penalty" grounding check — the backend README has claimed
since before this suite existed that it "runs extracted claims through a
cross-encoder to verify that the LLM output is 100% grounded in the
original source sentence."

Being direct about scope: there is no dedicated hallucination/grounding
checker anywhere in the production pipeline today. What DOES exist is a
real cross-encoder (cross-encoder/nli-deberta-v3-small, loaded via
get_nli_classifier() in clustering.py) already used in production for
claim-vs-claim contradiction detection during event polarization scoring.

This file operationalizes "grounded" the same way an NLI model actually
can: a claim is grounded in its evidence_sentence if the model does NOT
classify (evidence, claim) as a contradiction — entailment is the strong
case, neutral is acceptable for a claim that merely narrows or rephrases
the evidence, contradiction means the claim asserts something the
evidence doesn't support. This is a real, currently-passing test against
real production inference code (classify_nli_relationship), not a mock —
just honestly scoped to what the cross-encoder can actually tell you,
rather than inventing a new grounding subsystem under the label "test".
"""

import pytest

from app.services.clustering import classify_nli_relationship


def is_grounded(clf, evidence_sentence: str, claim: str) -> bool:
    label = classify_nli_relationship(clf, evidence_sentence, claim)
    return label != "contradiction"


@pytest.mark.model
def test_claim_that_restates_its_evidence_is_grounded(nli_classifier):
    evidence = "Tesla filed for an IPO worth $75 billion on Monday, according to regulatory filings."
    claim = "Tesla filed for an IPO worth $75 billion."
    assert is_grounded(nli_classifier, evidence, claim)


@pytest.mark.model
def test_claim_that_contradicts_its_evidence_is_not_grounded(nli_classifier):
    evidence = "Tesla filed for an IPO worth $75 billion on Monday, according to regulatory filings."
    claim = "Tesla decided not to pursue an IPO."
    assert not is_grounded(nli_classifier, evidence, claim)


@pytest.mark.model
def test_claim_with_a_flipped_number_contradicts_its_evidence(nli_classifier):
    # The kind of hallucination this check exists to catch: right shape,
    # wrong number — an LLM inventing a figure not actually in the source.
    evidence = "The company reported revenue of $12.4 billion in the third quarter."
    claim = "The company reported revenue of $40 billion in the third quarter."
    assert not is_grounded(nli_classifier, evidence, claim)


@pytest.mark.model
def test_unrelated_claim_is_not_treated_as_grounded_entailment(nli_classifier):
    # Not a contradiction either — genuinely unrelated text should land
    # "neutral", which is_grounded() treats as acceptable (still not the
    # dangerous case), but the top label itself should not be entailment.
    evidence = "Tesla filed for an IPO worth $75 billion on Monday."
    claim = "The weather in Paris was mild this week."
    label = classify_nli_relationship(nli_classifier, evidence, claim)
    assert label != "entailment"


def test_classify_nli_relationship_returns_none_on_inference_failure():
    def broken_clf(_text):
        raise RuntimeError("model unavailable")

    assert classify_nli_relationship(broken_clf, "premise", "hypothesis") is None


def test_classify_nli_relationship_handles_flat_and_nested_output_shapes():
    # transformers' top_k=None output shape has varied between a flat
    # list of label dicts and a list-of-lists across versions — the
    # parser needs to handle both without crashing.
    def flat(_text):
        return [{"label": "Entailment", "score": 0.9}]

    def nested(_text):
        return [[{"label": "Contradiction", "score": 0.9}, {"label": "Neutral", "score": 0.1}]]

    assert classify_nli_relationship(flat, "a", "b") == "entailment"
    assert classify_nli_relationship(nested, "a", "b") == "contradiction"
