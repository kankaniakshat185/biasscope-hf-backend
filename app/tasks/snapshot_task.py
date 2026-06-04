import asyncio
import logging
from app.celery_app import celery_app
from app.prisma_client import Prisma, Json

logger = logging.getLogger(__name__)

async def generate_snapshots_async():
    prisma = Prisma()
    await prisma.connect()

    from datetime import datetime, timedelta
    from app.services.ingestion import ingest_articles
    from app.services.cleaning import clean_and_deduplicate
    from app.services.nlp import analyze_articles
    from app.services.extraction import process_and_store_claims
    from app.services.clustering import run_claim_clustering, run_event_detection
    from app.main import get_search_intelligence

    try:
        subscriptions = await prisma.topicsubscription.find_many(where={"isActive": True})
        logger.info(f"Found {len(subscriptions)} active topic subscriptions.")

        for sub in subscriptions:
            topic = sub.topic
            
            # Default to 7 days ago if no snapshot exists
            last_date = sub.lastSnapshotAt or (datetime.utcnow() - timedelta(days=7))
            from_date_str = last_date.strftime("%Y-%m-%d")
            
            logger.info(f"Generating snapshot for topic: '{topic}' since {from_date_str}")
            
            # 1. Delta Ingestion
            raw_articles = await ingest_articles(query=topic, category="news", from_date=from_date_str)
            if not raw_articles:
                logger.info(f"No new articles for '{topic}'. Skipping.")
                continue
                
            cleaned_articles, _ = clean_and_deduplicate(raw_articles)
            analyzed_articles = analyze_articles(cleaned_articles)
            
            if not analyzed_articles:
                continue

            # 2. Store in DB to create the graph linkage
            search_record = await prisma.search.create(
                data={
                    "query": topic,
                    "category": "snapshot",
                    "userId": sub.userId
                }
            )
            
            article_records = []
            for art in analyzed_articles:
                record = await prisma.article.create(
                    data={
                        "searchId": search_record.id,
                        "title": art["title"],
                        "content": art.get("content", ""),
                        "source": art["source"],
                        "url": art["url"],
                        "publishedAt": datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00")) if art.get("publishedAt") else None,
                        "sentiment": art.get("sentiment"),
                        "sentimentScore": art.get("sentimentScore"),
                        "biasLabel": art.get("biasLabel"),
                        "sourceBias": art.get("sourceBias"),
                    }
                )
                article_records.append(record)

            # 3. Extraction Phase
            for art in article_records:
                if art.content:
                    await process_and_store_claims(
                        prisma, art.id, art.content, art.source, art.url, art.publishedAt, topic, art.title
                    )

            # 4. Clustering & Event Detection
            await run_claim_clustering(prisma)
            await run_event_detection(prisma)

            # 5. Build Snapshot Data
            # We fetch intelligence for THIS delta run specifically, or we can fetch for the whole topic.
            # For longitudinal drift, we want the current snapshot to represent the entire history up to now.
            # So we find ALL articles for this topic.
            all_searches = await prisma.search.find_many(where={"query": topic})
            search_ids = [s.id for s in all_searches]
            
            all_articles = await prisma.article.find_many(where={"searchId": {"in": search_ids}})
            article_ids = [a.id for a in all_articles]
            
            if not article_ids:
                continue

            evidence_records = await prisma.evidence.find_many(
                where={"articleId": {"in": article_ids}},
                include={"claim": True}
            )
            claim_ids = list(set([e.claimId for e in evidence_records if e.claimId]))
            claims = await prisma.claim.find_many(
                where={"id": {"in": claim_ids}},
                include={
                    "evidence": True,
                    "cluster": {"include": {"event": True}}
                }
            )
            
            # Aggregate stats
            events = set()
            bias_dist = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
            source_dist = {}
            
            for art in all_articles:
                b = art.biasLabel or "CENTER"
                if b in bias_dist:
                    bias_dist[b] += 1
                source_dist[art.source] = source_dist.get(art.source, 0) + 1

            for c in claims:
                if c.cluster and c.cluster.event:
                    events.add(c.cluster.event.id)

            total_left = bias_dist["LEFT"]
            total_right = bias_dist["RIGHT"]
            total_center = bias_dist["CENTER"]
            total_bias = total_left + total_right + total_center
            
            polarization = 0.0
            if total_bias > 0:
                polarization = (total_left + total_right) / total_bias

            # Create snapshot
            snapshot = await prisma.topicsnapshot.create(
                data={
                    "subscriptionId": sub.id,
                    "topic": topic,
                    "articleCount": len(all_articles),
                    "claimCount": len(claims),
                    "eventCount": len(events),
                    "polarizationIndex": polarization,
                    "biasDistribution": Json(bias_dist),
                    "sourceDistribution": Json(source_dist),
                }
            )
            
            # Save top events to snapshot
            # ... we would rank and link SnapshotEvent here
            
            await prisma.topicsubscription.update(
                where={"id": sub.id},
                data={"lastSnapshotAt": datetime.utcnow()}
            )
            
            logger.info(f"Finished delta processing and created snapshot for {topic}")

    except Exception as e:
        logger.error(f"Error in snapshot generation: {e}")
    finally:
        await prisma.disconnect()

@celery_app.task(name="app.tasks.snapshot_task.run_weekly_snapshots")
def run_weekly_snapshots():
    """
    Celery Beat task triggered weekly to compute drift and snapshots.
    """
    logger.info("Starting weekly snapshot generation job...")
    asyncio.run(generate_snapshots_async())
    logger.info("Weekly snapshot job complete.")
