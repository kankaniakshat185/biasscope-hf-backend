"""app/utils/create_vector_index.py — the D2 fix: adds the HNSW index
that was confirmed absent from the only migration file."""

import pytest

from app.utils import create_vector_index as index_module
from tests.fakes import FakePrisma


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(index_module, "prisma", prisma)
    return prisma


async def test_creates_an_hnsw_index_using_cosine_ops(fake_prisma):
    await index_module.main()

    fake_prisma.connect.assert_awaited_once()
    sql = fake_prisma.execute_raw.call_args.args[0]
    assert "hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert '"claim"' in sql
    fake_prisma.disconnect.assert_awaited_once()


async def test_disconnects_even_if_index_creation_fails(fake_prisma):
    fake_prisma.execute_raw.side_effect = RuntimeError("relation already has an index")

    with pytest.raises(RuntimeError):
        await index_module.main()

    fake_prisma.disconnect.assert_awaited_once()
