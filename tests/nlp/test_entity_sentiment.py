"""app/services/nlp.py's extract_entity_sentiment — rolls per-article
entity mentions into a cross-article knowledge graph of avg sentiment by
political lean."""

from app.services.nlp import extract_entity_sentiment


def test_averages_sentiment_per_bias_label_for_a_shared_entity():
    articles = [
        {"entities": {"Elon Musk": "PERSON"}, "bias_label": "LEFT", "sentiment_score": -0.8},
        {"entities": {"Elon Musk": "PERSON"}, "bias_label": "LEFT", "sentiment_score": -0.6},
        {"entities": {"Elon Musk": "PERSON"}, "bias_label": "RIGHT", "sentiment_score": 0.7},
        {"entities": {"Elon Musk": "PERSON"}, "bias_label": "RIGHT", "sentiment_score": 0.9},
    ]
    graph = extract_entity_sentiment(articles)
    assert graph["Elon Musk"]["avg_left_sentiment"] == -0.7
    assert graph["Elon Musk"]["avg_right_sentiment"] == 0.8
    assert graph["Elon Musk"]["mentions"] == 4


def test_entities_mentioned_only_once_are_excluded():
    articles = [{"entities": {"Obscure Person": "PERSON"}, "bias_label": "CENTER", "sentiment_score": 0.1}]
    graph = extract_entity_sentiment(articles)
    assert "Obscure Person" not in graph


def test_empty_articles_produces_empty_graph():
    assert extract_entity_sentiment([]) == {}
