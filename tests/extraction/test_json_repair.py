"""app/services/extraction.py's repair_truncated_json — LLMs sometimes cut
off mid-response when they hit max_tokens; this is what salvages a
still-useful partial claims list instead of discarding the whole
extraction."""

import json

from app.services.extraction import repair_truncated_json


def test_returns_valid_json_unchanged():
    raw = '{"claims": [{"text": "A."}]}'
    assert json.loads(repair_truncated_json(raw)) == json.loads(raw)


def test_repairs_truncated_object_missing_closing_brackets():
    raw = '{"claims": [{"text": "A."}, {"text": "B."}'  # cut off mid-second-object
    repaired = repair_truncated_json(raw)
    parsed = json.loads(repaired)
    assert parsed["claims"][0]["text"] == "A."


def test_combines_multiple_concatenated_json_objects():
    # Some models emit back-to-back JSON objects instead of one array.
    raw = '{"claims": [{"text": "A."}]}{"claims": [{"text": "B."}]}'
    parsed = json.loads(repair_truncated_json(raw))
    texts = [c["text"] for c in parsed["claims"]]
    assert texts == ["A.", "B."]


def test_returns_empty_string_for_unsalvageable_garbage():
    assert repair_truncated_json("not json at all, just prose") == ""


def test_empty_input_returns_empty_string():
    assert repair_truncated_json("") == ""
