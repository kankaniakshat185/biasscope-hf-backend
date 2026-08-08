"""app/services/ingestion.py's normalize_domains — turns whatever a user
typed into the advanced-search "restrict to these domains" field into
NewsAPI's expected bare-domain format."""

from app.services.ingestion import normalize_domains


def test_strips_scheme_and_www():
    assert normalize_domains("https://www.wsj.com/") == "wsj.com"


def test_bare_domain_passes_through():
    assert normalize_domains("wsj.com") == "wsj.com"


def test_handles_multiple_comma_separated_domains():
    assert normalize_domains("wsj.com, reuters.com") == "wsj.com,reuters.com"


def test_keeps_three_labels_for_known_multi_part_suffixes():
    assert normalize_domains("bbc.co.uk") == "bbc.co.uk"
    assert normalize_domains("https://www.bbc.co.uk/news") == "bbc.co.uk"


def test_strips_path_and_query_string():
    assert normalize_domains("https://www.nytimes.com/section/world?x=1") == "nytimes.com"


def test_collapses_a_deep_subdomain_to_the_registrable_domain():
    assert normalize_domains("edition.cnn.com") == "cnn.com"
