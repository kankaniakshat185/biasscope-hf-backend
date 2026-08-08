"""app/utils/reset_claim_graph.py — reset_claim_graph() must preserve
user accounts AND run history (Search/Article/Insight); reset_all() wipes
everything. Getting these two mixed up would be a very bad day."""

import pytest

from app.utils import reset_claim_graph as reset_module
from tests.fakes import FakePrisma


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(reset_module, "Prisma", lambda: prisma)
    return prisma


async def test_reset_claim_graph_wipes_the_intelligence_pipeline_only(fake_prisma):
    await reset_module.reset_claim_graph()

    fake_prisma.evidence.delete_many.assert_awaited_once_with(where={})
    fake_prisma.claim.delete_many.assert_awaited_once_with(where={})
    fake_prisma.claimcluster.delete_many.assert_awaited_once_with(where={})
    fake_prisma.event.delete_many.assert_awaited_once_with(where={})
    fake_prisma.llmcache.delete_many.assert_awaited_once_with(where={})
    fake_prisma.llmusage.delete_many.assert_awaited_once_with(where={})

    # Must NOT touch run history or accounts.
    fake_prisma.article.delete_many.assert_not_called()
    fake_prisma.search.delete_many.assert_not_called()
    fake_prisma.insight.delete_many.assert_not_called()
    fake_prisma.user.delete_many.assert_not_called()


async def test_reset_all_also_wipes_run_history(fake_prisma):
    await reset_module.reset_all()

    fake_prisma.claim.delete_many.assert_awaited_once_with(where={})
    fake_prisma.article.delete_many.assert_awaited_once_with(where={})
    fake_prisma.insight.delete_many.assert_awaited_once_with(where={})
    fake_prisma.search.delete_many.assert_awaited_once_with(where={})

    # Even reset_all leaves user accounts alone — no reset path touches auth data.
    fake_prisma.user.delete_many.assert_not_called()


async def test_reset_claim_graph_disconnects_even_on_failure(fake_prisma):
    fake_prisma.claim.delete_many.side_effect = RuntimeError("db error")
    await reset_module.reset_claim_graph()  # errors are caught and logged, not raised
    fake_prisma.disconnect.assert_awaited_once()
