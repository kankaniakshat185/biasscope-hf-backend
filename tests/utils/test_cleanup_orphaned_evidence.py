"""app/utils/cleanup_orphaned_evidence.py — the D1 prerequisite script:
must report/delete Evidence rows with no matching Article before the
Evidence.articleId FK constraint can be applied without failing."""

import pytest

from app.utils import cleanup_orphaned_evidence as cleanup_module
from tests.fakes import FakePrisma


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(cleanup_module, "prisma", prisma)
    return prisma


async def test_find_orphaned_evidence_count_reads_the_query_result(fake_prisma):
    fake_prisma.query_raw.return_value = [{"cnt": 7}]
    assert await cleanup_module.find_orphaned_evidence_count() == 7


async def test_find_orphaned_evidence_count_defaults_to_zero_on_empty_result(fake_prisma):
    fake_prisma.query_raw.return_value = []
    assert await cleanup_module.find_orphaned_evidence_count() == 0


async def test_delete_orphaned_evidence_returns_the_deleted_row_count(fake_prisma):
    fake_prisma.query_raw.return_value = [{"id": "e1"}, {"id": "e2"}]
    assert await cleanup_module.delete_orphaned_evidence() == 2


async def test_main_dry_run_reports_but_does_not_delete(fake_prisma, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cleanup_orphaned_evidence.py"])
    # First call (count check) finds orphans; a second DELETE call must never happen.
    fake_prisma.query_raw.return_value = [{"cnt": 3}]

    await cleanup_module.main()

    fake_prisma.connect.assert_awaited_once()
    fake_prisma.disconnect.assert_awaited_once()
    # Only the COUNT query ran — one call total, not a second DELETE.
    assert fake_prisma.query_raw.await_count == 1


async def test_main_with_delete_flag_actually_deletes(fake_prisma, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cleanup_orphaned_evidence.py", "--delete"])
    # First call: count (finds some). Second call: the DELETE itself.
    fake_prisma.query_raw.side_effect = [[{"cnt": 3}], [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]]

    await cleanup_module.main()

    assert fake_prisma.query_raw.await_count == 2


async def test_main_with_zero_orphans_never_checks_the_delete_flag(fake_prisma, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cleanup_orphaned_evidence.py", "--delete"])
    fake_prisma.query_raw.return_value = [{"cnt": 0}]

    await cleanup_module.main()

    # Safe to apply the FK constraint immediately — only the count query ran.
    assert fake_prisma.query_raw.await_count == 1
