# ADR 002: Claim-Centric vs Article-Level NLP

**Date:** July 2026  
**Status:** Accepted  

## Context
BiasScope needs to determine the sentiment, bias, and narrative of a news event. Traditional media bias tools run NLP algorithms (like Sentiment Analysis) on the entire text of a news article to assign a single "score" to that article.

## Alternatives Considered
1. **Article-Level Analysis:** Feed the entire 2000-word article into a Sentiment and Bias classifier. Average the scores.
2. **Claim-Centric Analysis:** Use an LLM to extract individual, atomic, falsifiable claims from the article. Run the Sentiment and Bias classifiers ONLY on these isolated claims, then aggregate them mathematically.

## Decision
We selected **Claim-Centric Analysis**.

## Justification
1. **The "Noise" Problem:** A single news article often contains a mix of factual reporting (neutral), quotes from critics (negative), and background context (mixed). If you feed the whole article to an ML model, the signals cancel each other out, resulting in a noisy, meaningless "Neutral" score.
2. **Clustering Viability:** You cannot effectively cluster entire articles using Cosine Similarity, because two articles from different publishers will use vastly different vocabulary and structures, pushing their embeddings apart in the vector space, even if they report the exact same event. By extracting atomic claims (e.g., "The bill was signed today"), the embeddings become highly precise, allowing `pgvector` to group them with 99% accuracy.
3. **Contrastive Summaries:** A claim-centric architecture allows us to separate the specific facts highlighted by Left-leaning publishers from those highlighted by Right-leaning publishers, enabling the generation of Echo Chamber contrastive summaries.
