"""
Test doubles for the Prisma client.

Nothing in this repo runs against a real Postgres/pgvector database in CI
or in this test suite — there isn't one available. Rather than skip
everything that touches the database, each "table" on FakePrisma is an
AsyncMock: tests configure `.find_unique.return_value = fake_claim(...)`
etc. and can assert on `.update.call_args` afterward, so the actual
business logic in app/services and app/routers gets exercised and
verified even though no real query ever runs. This is a deliberate
trade-off, not an oversight — see AUDIT_TASKS.md T1 for why full
DB-backed integration tests aren't in scope here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

PRISMA_MODELS = [
    "search", "article", "insight", "claim", "evidence", "event",
    "claimcluster", "topicsubscription", "topicsnapshot", "demosnapshot",
    "llmcache", "llmusage", "session", "user",
]


def _dump(value):
    if isinstance(value, FakeRecord):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


class FakeRecord(SimpleNamespace):
    """Stands in for a Prisma model instance. Supports `.model_dump()`
    (used by app/services/intelligence.py's get_results, matching the
    real pydantic-generated models) recursively across nested relations."""

    def model_dump(self, mode=None):
        return {k: _dump(v) for k, v in vars(self).items()}


class FakePrisma:
    """One AsyncMock-backed proxy per Prisma model, plus query_raw/
    execute_raw/connect/disconnect. Default return values are the
    "nothing found" shape (None / [] / 0) so a test only needs to
    override what it actually cares about."""

    def __init__(self):
        for model in PRISMA_MODELS:
            setattr(self, model, MagicMock(
                find_unique=AsyncMock(return_value=None),
                find_first=AsyncMock(return_value=None),
                find_many=AsyncMock(return_value=[]),
                create=AsyncMock(),
                create_many=AsyncMock(),
                update=AsyncMock(),
                delete=AsyncMock(),
                delete_many=AsyncMock(),
                upsert=AsyncMock(),
                count=AsyncMock(return_value=0),
            ))
        self.query_raw = AsyncMock(return_value=[])
        self.execute_raw = AsyncMock(return_value=0)
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()


# ── Record builders — one per model, matching prisma/schema.prisma ──────

def fake_search(id="search-1", userId="user-1", query="elon musk", category="Technology",
                 phase2Status="pending", **kw):
    return FakeRecord(id=id, userId=userId, query=query, category=category,
                       createdAt="2026-01-01T00:00:00Z", phase2Status=phase2Status,
                       articles=kw.pop("articles", []), insights=kw.pop("insights", []), **kw)


def fake_article(id="article-1", searchId="search-1", title="Headline", content="Body text",
                  source="reuters.com", url="https://reuters.com/a", **kw):
    return FakeRecord(
        id=id, searchId=searchId, title=title, content=content, source=source, url=url,
        publishedAt=None, sentiment="neutral", sentimentScore=0.0, biasLabel="UNKNOWN",
        sourceBias="UNKNOWN", deviationScore=0.0, entities={}, createdAt="2026-01-01T00:00:00Z",
        **kw,
    )


def fake_evidence(id="ev-1", claimId="claim-1", articleId="article-1", sentence="They said so.",
                   source="reuters.com", url="https://reuters.com/a", stance="MENTION", **kw):
    return FakeRecord(id=id, claimId=claimId, articleId=articleId, sentence=sentence,
                       source=source, url=url, publishedAt="2026-01-01T00:00:00Z", stance=stance, **kw)


def fake_claim(id="claim-1", canonicalClaim="Tesla filed for an IPO.", claimType="EVENT",
                qualityScore=0.8, confidence=0.9, clusterId=None, cluster=None, evidence=None, **kw):
    return FakeRecord(
        id=id, canonicalClaim=canonicalClaim, claimType=claimType, qualityScore=qualityScore,
        confidence=confidence, consensusScore=0.0, contradictionScore=0.0,
        createdAt="2026-01-01T00:00:00Z", clusterId=clusterId, cluster=cluster,
        evidence=evidence if evidence is not None else [], **kw,
    )


def fake_cluster(id="cluster-1", title="Tesla IPO Filing", canonicalClaim="Tesla filed for an IPO.",
                  consensusScore=0.5, cohesionScore=0.8, eventId=None, event=None, claims=None, **kw):
    return FakeRecord(
        id=id, title=title, canonicalClaim=canonicalClaim, consensusScore=consensusScore,
        cohesionScore=cohesionScore, eventId=eventId, event=event,
        claims=claims if claims is not None else [], **kw,
    )


def fake_event(id="event-1", title="Tesla IPO", description="Tesla filed for an IPO.",
               importanceScore=0.7, claimClusters=None, **kw):
    return FakeRecord(id=id, title=title, description=description, importanceScore=importanceScore,
                       createdAt="2026-01-01T00:00:00Z",
                       claimClusters=claimClusters if claimClusters is not None else [], **kw)


def fake_session(id="session-1", token="tok_abc123", userId="user-1", expires_in_future=True, **kw):
    from datetime import datetime, timedelta, timezone
    delta = timedelta(hours=1) if expires_in_future else timedelta(hours=-1)
    return FakeRecord(id=id, token=token, userId=userId,
                       expiresAt=datetime.now(timezone.utc) + delta,
                       createdAt=datetime.now(timezone.utc), updatedAt=datetime.now(timezone.utc),
                       ipAddress=None, userAgent=None, **kw)


def fake_request(cookies=None, headers=None):
    """A minimal stand-in for fastapi.Request — app/deps/auth.py only
    reads .cookies and .headers.get(...)."""
    cookies = cookies or {}
    headers = headers or {}
    return SimpleNamespace(
        cookies=cookies,
        headers=SimpleNamespace(get=lambda k, default=None: headers.get(k.lower(), default)),
    )
