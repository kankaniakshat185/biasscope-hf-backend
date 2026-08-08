"""app/services/extraction.py's compute_quality_score — the heuristic gate
that rejects opinion/biographical/commentary content before it ever
reaches the database. Pure function, no LLM/DB/model involved."""

from app.services.extraction import compute_quality_score


def test_rejects_questions():
    assert compute_quality_score("Did Tesla really file for an IPO?") == 0.0


def test_rejects_opinion_language():
    assert compute_quality_score("The policy is deeply controversial among voters.") == 0.0


def test_rejects_journalist_commentary():
    assert compute_quality_score("Critics argue the bill raises questions about oversight.") == 0.0


def test_heavily_penalizes_biographical_content():
    score = compute_quality_score("Musk married his partner and has several children.")
    assert score <= 0.10


def test_accepts_a_solid_event_claim():
    score = compute_quality_score("Tesla filed for an IPO worth $75 billion on Monday.")
    assert score >= 0.40


def test_accepts_a_solid_numeric_claim():
    score = compute_quality_score("The company reported revenue of $12.4 billion in Q3.")
    assert score >= 0.40


# --- Q1 regressions: word-boundary matching, not bare substring ---
# These specifically guard the fix in AUDIT_TASKS.md Q1 — before it, each
# of these was wrongly rejected because the signal word appeared as a
# substring of a longer, unrelated word.

def test_does_not_reject_children_as_a_substring_of_grandchildren():
    score = compute_quality_score("The senator's grandchildren attended the ceremony downtown.")
    # Still may not score high (no numbers/action verbs), but must not hit
    # the biographical hard-cap of 0.10 the way "children" substring-matching would.
    assert score > 0.10


def test_does_not_reject_damaged_as_substring_of_undamaged():
    score = compute_quality_score("The bridge remained undamaged after the flooding event.")
    assert score != 0.0


def test_does_not_reject_controversial_as_substring_of_uncontroversial():
    score = compute_quality_score("The uncontroversial measure passed unanimously on Tuesday.")
    assert score != 0.0


def test_still_rejects_controversial_as_a_whole_word():
    assert compute_quality_score("The controversial measure passed narrowly on Tuesday.") == 0.0
