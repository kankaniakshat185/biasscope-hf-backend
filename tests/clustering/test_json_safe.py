"""app/services/clustering.py's parse_json_safe — same spirit as
extraction's repair_truncated_json but for the simpler canonical-
claim/event-summary LLM responses."""

from app.services.clustering import parse_json_safe


def test_parses_valid_json():
    assert parse_json_safe('{"canonical_claim": "X happened."}') == {"canonical_claim": "X happened."}


def test_returns_fallback_dict_for_invalid_json():
    assert parse_json_safe("not json", fallback={"a": 1}) == {"a": 1}


def test_returns_empty_dict_by_default_for_invalid_json():
    assert parse_json_safe("not json") == {}


def test_empty_string_returns_fallback():
    assert parse_json_safe("", fallback={"x": 1}) == {"x": 1}


def test_none_like_empty_input_returns_empty_dict_by_default():
    assert parse_json_safe("") == {}
