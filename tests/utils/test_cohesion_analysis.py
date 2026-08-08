"""app/utils/cohesion_analysis.py — CLI diagnostic table. Tested by
capturing stdout, since printing the report IS the entire behavior."""

import pytest

from app.utils import cohesion_analysis as cohesion_module
from tests.fakes import FakePrisma, fake_claim, fake_cluster, fake_evidence


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(cohesion_module, "Prisma", lambda: prisma)
    return prisma


async def test_prints_a_row_per_cluster_with_source_count_and_cohesion(fake_prisma, capsys):
    cluster = fake_cluster(
        title="Tesla IPO Filing", cohesionScore=0.83, eventId="event-1",
        claims=[fake_claim(evidence=[fake_evidence(source="reuters.com"), fake_evidence(source="apnews.com")])],
    )
    fake_prisma.claimcluster.find_many.return_value = [cluster]

    await cohesion_module.analyze_cohesion()

    output = capsys.readouterr().out
    assert "Tesla IPO Filing" in output
    assert "0.830" in output
    assert "Yes" in output  # has an eventId


async def test_handles_a_cluster_with_no_claims_without_crashing(fake_prisma, capsys):
    fake_prisma.claimcluster.find_many.return_value = [fake_cluster(claims=[], eventId=None)]

    await cohesion_module.analyze_cohesion()

    output = capsys.readouterr().out
    assert "No" in output  # no eventId -> not accepted as an event


async def test_disconnects_after_reporting(fake_prisma):
    fake_prisma.claimcluster.find_many.return_value = []
    await cohesion_module.analyze_cohesion()
    fake_prisma.connect.assert_awaited_once()
    fake_prisma.disconnect.assert_awaited_once()
