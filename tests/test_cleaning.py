"""app/services/cleaning.py — dedup and sanitization before anything else
in the pipeline sees an article."""

from app.services.cleaning import clean_and_deduplicate


def test_drops_articles_with_empty_title():
    raw = [
        {"url": "https://a.com/1", "title": "", "content": "..."},
        {"url": "https://a.com/2", "title": "Real Headline", "content": "..."},
    ]
    cleaned, dupes_removed = clean_and_deduplicate(raw)
    assert [a["url"] for a in cleaned] == ["https://a.com/2"]
    assert dupes_removed == 1


def test_drops_exact_duplicate_urls():
    raw = [
        {"url": "https://a.com/1", "title": "Headline One", "content": "..."},
        {"url": "https://a.com/1", "title": "Headline One (again)", "content": "..."},
    ]
    cleaned, dupes_removed = clean_and_deduplicate(raw)
    assert len(cleaned) == 1
    assert dupes_removed == 1


def test_drops_near_duplicate_titles_across_different_urls():
    # Same story, two different outlets rewording the same headline —
    # exactly the case this fuzzy-match gate exists to catch.
    raw = [
        {"url": "https://a.com/1", "title": "Musk Announces New Tesla Factory in Texas", "content": "..."},
        {"url": "https://b.com/1", "title": "Musk announces new Tesla factory in Texas", "content": "..."},
    ]
    cleaned, dupes_removed = clean_and_deduplicate(raw)
    assert len(cleaned) == 1
    assert dupes_removed == 1


def test_keeps_genuinely_distinct_articles():
    raw = [
        {"url": "https://a.com/1", "title": "Musk Announces New Tesla Factory in Texas", "content": "..."},
        {"url": "https://b.com/1", "title": "Senate Passes Infrastructure Bill", "content": "..."},
        {"url": "https://c.com/1", "title": "Local Weather Forecast for the Weekend", "content": "..."},
    ]
    cleaned, dupes_removed = clean_and_deduplicate(raw)
    assert len(cleaned) == 3
    assert dupes_removed == 0


def test_strips_null_bytes_from_content_and_title():
    raw = [{"url": "https://a.com/1", "title": "Headline\x00", "content": "Body\x00text"}]
    cleaned, _ = clean_and_deduplicate(raw)
    assert "\x00" not in cleaned[0]["title"]
    assert "\x00" not in cleaned[0]["content"]


def test_does_not_mutate_the_caller_input_dicts():
    # R12 regression: this used to mutate `article` (a raw_articles element)
    # in place and return the SAME object reference in `cleaned` — so
    # raw_articles ended up silently carrying the cleaned/stripped values
    # too. Nothing currently re-reads raw_articles after this call, but a
    # future change that did would see mutated data under a "raw" name.
    raw = [{"url": "https://a.com/1", "title": "Headline\x00", "content": "Body\x00text"}]
    cleaned, _ = clean_and_deduplicate(raw)
    assert cleaned[0] is not raw[0]
    assert raw[0]["title"] == "Headline\x00"
    assert raw[0]["content"] == "Body\x00text"
