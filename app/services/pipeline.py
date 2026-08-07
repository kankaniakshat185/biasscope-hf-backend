"""
Search pipeline orchestration.

This owns the actual "run a search" business logic (ingest → clean → NLP →
validate → narrative → persist) and the Phase 2 background pipeline
(extraction → clustering → event detection). It used to live directly
inside the `/search` route handler in app/main.py — moving it here means
the orchestration can be tested, reused (e.g. by the weekly snapshot task,
which currently reimplements a slightly different version of the same
steps), and read without wading through routing/response-model code.
"""

import logging
import os

from fastapi import HTTPException

from ..db import prisma
from ..prisma_client import Json
from .cleaning import clean_and_deduplicate
from .clustering import run_claim_clustering, run_event_detection
from .extraction import process_and_store_claims
from .ingestion import ingest_articles
from .nlp import (
    analyze_articles,
    extract_entity_sentiment,
    generate_contrastive_summaries,
    generate_narrative,
    get_source_reliability,
)
from .validation import validate_articles

logger = logging.getLogger(__name__)


async def run_search_pipeline(
    query: str,
    category: str,
    domains: str | None,
    exclude_domains: str | None,
    from_date: str | None,
    to_date: str | None,
    user_id: str | None,
) -> dict:
    """Runs the synchronous half of a search: ingest, analyze, validate,
    summarize, and persist the Search/Article/Insight rows. Returns the
    created search id. Raises HTTPException (via ingest/validate) on
    unrecoverable failures — callers should let those propagate to FastAPI.
    """
    logger.info(f"Starting search for: {query} in {category}")

    # 1. Ingestion
    raw_articles = await ingest_articles(
        query=query,
        category=category,
        domains=domains,
        exclude_domains=exclude_domains,
        from_date=from_date,
        to_date=to_date,
    )
    if not raw_articles:
        raise HTTPException(
            status_code=404,
            detail=(
                f"NewsAPI error or No articles found for: query='{query}', "
                f"domains='{domains}', fromDate='{from_date}'. Please ensure "
                "domains are properly formatted (e.g. 'wsj.com') and dates "
                "are within the last 30 days."
            ),
        )

    # 2. Cleaning
    cleaned_articles, dupes_removed = clean_and_deduplicate(raw_articles)

    # 3. NLP
    analyzed_articles = analyze_articles(cleaned_articles)

    # 4. Validation
    validation_metrics = validate_articles(analyzed_articles)

    valid_articles_list = validation_metrics.get("valid_articles_list", [])
    if not valid_articles_list:
        raise HTTPException(status_code=404, detail="All scraped articles failed validation or were empty.")

    validation_metrics["total_articles"] = len(raw_articles)
    validation_metrics["duplicates_removed"] = dupes_removed

    # 5. Narrative Gen & Contrastive Echo Chambers & Entity Extraction
    summary = await generate_narrative(prisma, analyzed_articles)
    contrastive_summaries = await generate_contrastive_summaries(prisma, analyzed_articles)
    entity_sentiment_graph = extract_entity_sentiment(analyzed_articles)

    # Source Reliability Confidence (using shared function)
    if valid_articles_list:
        reliability_scores = []
        cred_counts = {"High": 0, "Medium": 0, "Low": 0, "Mixed": 0, "Unknown": 0}
        for a in valid_articles_list:
            score, tier = get_source_reliability(a.get("source", ""))
            reliability_scores.append(score)
            cred_counts[tier] = cred_counts.get(tier, 0) + 1
        avg_confidence = sum(reliability_scores) / len(reliability_scores)
    else:
        avg_confidence = 0.0
        cred_counts = {"High": 0, "Medium": 0, "Low": 0, "Mixed": 0, "Unknown": 0}

    drift_metrics = {
        "source_reliability_confidence": avg_confidence,
        "credibility_breakdown": cred_counts,
    }

    # 6. Store to DB
    search_record = await prisma.search.create(
        data={
            "query": query,
            "category": category,
            "userId": user_id,
        }
    )

    # Insert articles
    article_creates = []
    for art in valid_articles_list:
        article_creates.append({
            "searchId": search_record.id,
            "title": art.get("title", "No Title"),
            "content": art.get("content", ""),
            "source": art.get("source", "Unknown"),
            "url": art.get("url", ""),
            "sentiment": art.get("sentiment", "neutral"),
            "sentimentScore": float(art.get("sentiment_score", 0.0)),
            "biasLabel": art.get("bias_label", "UNKNOWN"),
            "sourceBias": art.get("source_bias", "UNKNOWN"),
            "deviationScore": float(art.get("deviation_score", 0.0)),
            "entities": Json(art.get("entities", {})),
            "publishedAt": art.get("published_at"),
        })

    if article_creates:
        # Every prisma call in this codebase builds its `data=`/`where=`/
        # `include=` argument as a plain dict literal rather than the
        # generated per-model TypedDict (ArticleCreateWithoutRelationsInput
        # here) — consistent throughout, and not worth a one-off exception
        # just for this call site.
        await prisma.article.create_many(data=article_creates)  # type: ignore[arg-type]

    # Insert Insights
    await prisma.insight.create(
        data={
            "search": {"connect": {"id": search_record.id}},
            "avgSentiment": float(validation_metrics.get("avg_sentiment", 0.0)),
            "topKeywords": Json(validation_metrics.get("top_keywords", [])),
            "biasDistribution": Json(validation_metrics.get("bias_distribution", {"LEFT": 0, "CENTER": 0, "RIGHT": 0, "UNKNOWN": 0})),
            "dataQualityScore": float(validation_metrics.get("data_quality_score", 0.0)),
            "polarizationScore": float(validation_metrics.get("polarization_score", 0.0)),
            "totalArticles": validation_metrics["total_articles"],
            "validArticles": validation_metrics["valid_articles"],
            "duplicatesRemoved": validation_metrics["duplicates_removed"],
            "missingContent": validation_metrics["missing_content"],
            "datasetMetrics": Json(validation_metrics.get("dataset_metrics", {})),
            "narrativeSummary": summary,
            "leftWingSummary": contrastive_summaries.get("left", ""),
            "rightWingSummary": contrastive_summaries.get("right", ""),
            "entitySentiment": Json(entity_sentiment_graph),
            "driftMetrics": Json(drift_metrics),
        }
    )

    return {"search_id": search_record.id, "message": "Search processed successfully."}


async def run_phase2_pipeline(search_id: str) -> None:
    """
    Feature 2: Pipeline Stage Toggles
    Set env vars to skip stages during development:
      SKIP_EXTRACTION=1  — reuse existing claims
      SKIP_CLUSTERING=1  — reuse existing clusters
      SKIP_EVENTS=1      — reuse existing events
    """
    logger.info(f"Starting background Phase 2 pipeline for search {search_id}...")
    try:
        search_record = await prisma.search.find_unique(where={"id": search_id})
        query = search_record.query if search_record else ""

        if not os.getenv("SKIP_EXTRACTION"):
            articles = await prisma.article.find_many(where={"searchId": search_id})
            for art in articles:
                if art.content:
                    await process_and_store_claims(
                        prisma, art.id, art.content, art.source, art.url,
                        art.publishedAt, query, art.title,
                    )
            logger.info(f"Extraction complete for search {search_id}")
        else:
            logger.info("SKIP_EXTRACTION=1 — reusing existing claims")

        if not os.getenv("SKIP_CLUSTERING"):
            # Scoped to this search's query — see the comment on
            # run_claim_clustering for why that's the fix, not a time window.
            await run_claim_clustering(prisma, query)
            logger.info("Clustering complete")
        else:
            logger.info("SKIP_CLUSTERING=1 — reusing existing clusters")

        if not os.getenv("SKIP_EVENTS"):
            await run_event_detection(prisma)
            logger.info("Event detection complete")
        else:
            logger.info("SKIP_EVENTS=1 — reusing existing events")

        logger.info("Phase 2 pipeline complete.")
    except Exception as e:
        logger.exception(f"Phase 2 pipeline error: {e}")
