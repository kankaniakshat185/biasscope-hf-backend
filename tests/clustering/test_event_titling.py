"""app/services/clustering.py's generate_event_title — zero-LLM-call
deterministic title generation (entity extraction + TF-IDF + action-word mapping)."""

from app.services.clustering import generate_event_title


def test_combines_entity_with_recognized_action_word():
    title = generate_event_title([
        "Tesla filed for an IPO worth $75 billion on Monday.",
        "Tesla's IPO filing values the company at a record high.",
    ])
    assert "Tesla" in title
    assert "IPO" in title or "Filing" in title


def test_falls_back_to_truncated_claim_text_with_no_recognizable_entities():
    title = generate_event_title(["something happened somewhere with no proper nouns at all today"])
    assert isinstance(title, str)
    assert len(title) > 0


def test_empty_input_returns_placeholder():
    assert generate_event_title([]) == "Unclassified Event"


def test_recognizes_acronyms_alongside_named_entities():
    title = generate_event_title([
        "The BJP announced a new policy on Monday in New Delhi.",
        "BJP leaders confirmed the policy will take effect next year.",
    ])
    assert "BJP" in title


def test_does_not_repeat_stopwords_as_the_title():
    title = generate_event_title(["The event happened on Monday according to reports."])
    assert title.strip() not in {"The", "Monday", "According"}
