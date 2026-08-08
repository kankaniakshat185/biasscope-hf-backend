"""app/services/nlp.py's extract_keywords — entity + TF-IDF keyword
discovery used both in validate_articles' top_keywords and the fallback
narrative."""

from app.services.nlp import extract_keywords


def test_extracts_capitalized_entities_when_no_ner_entities_present():
    articles = [
        {"title": "Tesla Announces New Factory", "content": "Tesla Motors confirmed the plan.", "entities": {}},
        {"title": "Tesla Stock Rises", "content": "Tesla shares rose 5 percent today.", "entities": {}},
    ]
    keywords = extract_keywords(articles)
    words = [k["word"] for k in keywords]
    assert any("Tesla" in w for w in words)


def test_prefers_ner_derived_entities_when_available():
    articles = [
        {"title": "A", "content": "irrelevant", "entities": {"Elon Musk": "PERSON"}},
        {"title": "B", "content": "irrelevant", "entities": {"Elon Musk": "PERSON"}},
    ]
    keywords = extract_keywords(articles)
    assert any(k["word"] == "Elon Musk" for k in keywords)


def test_returns_at_most_ten_keywords():
    articles = [
        {"title": f"Headline About TopicNumber{i}", "content": f"Content mentioning TopicNumber{i} repeatedly.", "entities": {}}
        for i in range(20)
    ]
    keywords = extract_keywords(articles)
    assert len(keywords) <= 10


def test_empty_article_list_does_not_crash():
    assert extract_keywords([]) == []
