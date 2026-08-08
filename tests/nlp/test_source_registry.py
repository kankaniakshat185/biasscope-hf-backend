"""app/services/nlp.py's SOURCE_RELIABILITY / SOURCE_BIAS_REGISTRY and
get_source_reliability() — this IS the single source of truth the A3 fix
made the frontend defer to instead of its own (deleted, disagreeing)
hardcoded domain arrays. A regression here is now user-visible on every
article card."""

import pytest

from app.services.nlp import SOURCE_BIAS_REGISTRY, SOURCE_RELIABILITY, get_source_reliability


@pytest.mark.parametrize("domain,expected_tier", [
    ("reuters.com", "High"),
    ("apnews.com", "High"),
    ("bbc.co.uk", "High"),
    ("cnn.com", "Medium"),
    ("breitbart.com", "Low"),
    ("infowars.com", "Low"),
    ("opindia.com", "Mixed"),
])
def test_known_domains_map_to_the_expected_tier(domain, expected_tier):
    _, tier = get_source_reliability(domain)
    assert tier == expected_tier


def test_unknown_domain_defaults_to_unknown_tier_with_neutral_score():
    score, tier = get_source_reliability("some-random-blog-that-does-not-exist.example")
    assert tier == "Unknown"
    assert score == 0.50


def test_matches_by_substring_so_full_urls_work_not_just_bare_domains():
    score, tier = get_source_reliability("https://www.reuters.com/world/some-article")
    assert tier == "High"


def test_is_case_insensitive():
    score_lower, tier_lower = get_source_reliability("reuters.com")
    score_upper, tier_upper = get_source_reliability("REUTERS.COM")
    assert (score_lower, tier_lower) == (score_upper, tier_upper)


def test_every_reliability_score_is_between_zero_and_one():
    assert all(0.0 <= score <= 1.0 for score in SOURCE_RELIABILITY.values())


def test_every_bias_registry_value_is_a_recognized_label():
    assert set(SOURCE_BIAS_REGISTRY.values()) <= {"LEFT", "CENTER", "RIGHT"}


def test_tier_boundaries_are_internally_consistent():
    # get_source_reliability's own bucket thresholds (>=0.85 High, >=0.65
    # Medium, >=0.45 Mixed, else Low) should agree with what every entry
    # in the registry actually gets bucketed as — catches someone adding a
    # score that silently lands in a different tier than intended.
    for domain, score in SOURCE_RELIABILITY.items():
        _, tier = get_source_reliability(domain)
        if score >= 0.85:
            assert tier == "High", domain
        elif score >= 0.65:
            assert tier == "Medium", domain
        elif score >= 0.45:
            assert tier == "Mixed", domain
        else:
            assert tier == "Low", domain
