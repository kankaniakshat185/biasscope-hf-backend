"""
app/tasks/snapshot_task.py's generate_snapshots_async — the weekly Celery
Beat job that delta-ingests new articles per active topic subscription.

Two things make this different from the other service tests: it
constructs its own `Prisma()` instance (not the shared `app.db.prisma`
singleton — a real Celery worker process legitimately wants its own), and
it imports its service dependencies LOCALLY inside the function body
rather than at module level. Both mean patching `app.tasks.snapshot_task`'s
own namespace doesn't work — the fake Prisma() has to replace the
`Prisma` class import, and the service functions have to be patched on
their SOURCE modules (the local `from X import Y` re-resolves `X.Y` at
call time, so a patch made before calling the function is still picked up).
"""

from unittest.mock import AsyncMock

import pytest

from app.tasks import snapshot_task as snapshot_task_module
from tests.fakes import FakePrisma, FakeRecord


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(snapshot_task_module, "Prisma", lambda: prisma)
    return prisma


@pytest.fixture
def mocked_stages(monkeypatch):
    ingest = AsyncMock(return_value=[])
    process_claims = AsyncMock()
    run_clustering = AsyncMock()
    run_events = AsyncMock()
    monkeypatch.setattr("app.services.ingestion.ingest_articles", ingest)
    monkeypatch.setattr("app.services.extraction.process_and_store_claims", process_claims)
    monkeypatch.setattr("app.services.clustering.run_claim_clustering", run_clustering)
    monkeypatch.setattr("app.services.clustering.run_event_detection", run_events)
    return {
        "ingest_articles": ingest,
        "process_and_store_claims": process_claims,
        "run_claim_clustering": run_clustering,
        "run_event_detection": run_events,
    }


def _subscription(id="sub-1", topic="elon musk", userId="user-1", lastSnapshotAt=None):
    return FakeRecord(id=id, topic=topic, userId=userId, lastSnapshotAt=lastSnapshotAt, isActive=True)


async def test_no_active_subscriptions_is_a_clean_no_op(fake_prisma, mocked_stages):
    fake_prisma.topicsubscription.find_many.return_value = []

    await snapshot_task_module.generate_snapshots_async()

    fake_prisma.search.create.assert_not_called()
    fake_prisma.disconnect.assert_awaited_once()


async def test_skips_a_topic_with_no_new_articles(fake_prisma, mocked_stages):
    fake_prisma.topicsubscription.find_many.return_value = [_subscription()]
    mocked_stages["ingest_articles"].return_value = []  # nothing new since last snapshot

    await snapshot_task_module.generate_snapshots_async()

    fake_prisma.search.create.assert_not_called()
    fake_prisma.topicsnapshot.create.assert_not_called()


async def test_processes_a_topic_with_new_articles_end_to_end(fake_prisma, mocked_stages, monkeypatch):
    fake_prisma.topicsubscription.find_many.return_value = [_subscription(topic="elon musk")]
    mocked_stages["ingest_articles"].return_value = [
        {"title": "T", "url": "https://reuters.com/a", "source": "reuters.com", "content": "..."},
    ]
    monkeypatch.setattr(
        "app.services.nlp.analyze_articles",
        lambda articles: [{**a, "sentiment": "neutral", "sentimentScore": 0.0, "biasLabel": "LEFT", "sourceBias": "LEFT"} for a in articles],
    )
    fake_prisma.search.create.return_value = FakeRecord(id="search-1")
    fake_prisma.article.create.return_value = FakeRecord(id="art-1", title="T", content="...", source="reuters.com", url="https://reuters.com/a", publishedAt=None)
    fake_prisma.search.find_many.return_value = [FakeRecord(id="search-1")]
    fake_prisma.article.find_many.return_value = [FakeRecord(id="art-1", biasLabel="LEFT", source="reuters.com")]
    fake_prisma.evidence.find_many.return_value = []

    await snapshot_task_module.generate_snapshots_async()

    fake_prisma.search.create.assert_awaited_once()
    create_kwargs = fake_prisma.search.create.call_args.kwargs["data"]
    assert create_kwargs["query"] == "elon musk"
    assert create_kwargs["userId"] == "user-1"

    mocked_stages["process_and_store_claims"].assert_awaited_once()
    # P1 fix applies here too: clustering scoped to this topic's query.
    mocked_stages["run_claim_clustering"].assert_awaited_once_with(fake_prisma, "elon musk")
    mocked_stages["run_event_detection"].assert_awaited_once()

    fake_prisma.topicsnapshot.create.assert_awaited_once()
    snapshot_kwargs = fake_prisma.topicsnapshot.create.call_args.kwargs["data"]
    assert snapshot_kwargs["articleCount"] == 1
    assert snapshot_kwargs["subscriptionId"] == "sub-1"

    fake_prisma.topicsubscription.update.assert_awaited_once()
    assert fake_prisma.topicsubscription.update.call_args.kwargs["where"] == {"id": "sub-1"}


async def test_polarization_index_reflects_bias_distribution(fake_prisma, mocked_stages, monkeypatch):
    fake_prisma.topicsubscription.find_many.return_value = [_subscription()]
    mocked_stages["ingest_articles"].return_value = [{"title": "T", "url": "u", "source": "s", "content": "..."}]
    monkeypatch.setattr("app.services.nlp.analyze_articles", lambda articles: articles)
    fake_prisma.search.create.return_value = FakeRecord(id="search-1")
    fake_prisma.article.create.return_value = FakeRecord(id="art-1", content=None, source="s", url="u", publishedAt=None)
    fake_prisma.search.find_many.return_value = [FakeRecord(id="search-1")]
    # 2 LEFT, 2 RIGHT, 0 CENTER -> fully polarized (no center ground truth)
    fake_prisma.article.find_many.return_value = [
        FakeRecord(id="a1", biasLabel="LEFT", source="s1"),
        FakeRecord(id="a2", biasLabel="LEFT", source="s2"),
        FakeRecord(id="a3", biasLabel="RIGHT", source="s3"),
        FakeRecord(id="a4", biasLabel="RIGHT", source="s4"),
    ]
    fake_prisma.evidence.find_many.return_value = []

    await snapshot_task_module.generate_snapshots_async()

    snapshot_kwargs = fake_prisma.topicsnapshot.create.call_args.kwargs["data"]
    assert snapshot_kwargs["polarizationIndex"] == 1.0


async def test_an_exception_mid_processing_is_caught_not_raised(fake_prisma, mocked_stages):
    fake_prisma.topicsubscription.find_many = AsyncMock(side_effect=RuntimeError("db exploded"))

    await snapshot_task_module.generate_snapshots_async()  # must not raise

    fake_prisma.disconnect.assert_awaited_once()


async def test_disconnects_even_when_an_exception_occurs(fake_prisma, mocked_stages):
    fake_prisma.topicsubscription.find_many.return_value = [_subscription()]
    mocked_stages["ingest_articles"].side_effect = RuntimeError("network blew up")

    await snapshot_task_module.generate_snapshots_async()

    fake_prisma.disconnect.assert_awaited_once()


def test_run_weekly_snapshots_invokes_the_async_pipeline(monkeypatch):
    called = {}

    def fake_run(coro):
        called["ran"] = True
        coro.close()  # avoid "coroutine was never awaited" warning

    monkeypatch.setattr(snapshot_task_module.asyncio, "run", fake_run)
    snapshot_task_module.run_weekly_snapshots()
    assert called.get("ran") is True
