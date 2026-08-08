"""app/routers/subscriptions.py — same identity rule as history.py:
everything is scoped to Depends(get_current_user_id), never a client-
supplied field."""

import pytest

from app.deps.auth import get_current_user_id
from app.routers import subscriptions as subs_module
from tests.fakes import FakePrisma, FakeRecord
from tests.routers.conftest import make_client


@pytest.fixture
def client():
    return make_client(subs_module.router)


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(subs_module, "prisma", prisma)
    return prisma


def _authed_as(app, user_id):
    app.dependency_overrides[get_current_user_id] = lambda: user_id


def test_post_401s_without_auth(client, fake_prisma, monkeypatch):
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())
    res = client.post("/subscriptions", json={"topic": "elon musk"})
    assert res.status_code == 401


def test_post_creates_subscription_for_the_caller_lowercased(client, fake_prisma):
    fake_prisma.topicsubscription.find_first.return_value = None
    fake_prisma.topicsubscription.create.return_value = FakeRecord(id="sub-1", userId="user-1", topic="elon musk")
    _authed_as(client.app, "user-1")

    res = client.post("/subscriptions", json={"topic": "Elon Musk"})

    assert res.status_code == 200
    create_kwargs = fake_prisma.topicsubscription.create.call_args.kwargs
    assert create_kwargs["data"] == {"userId": "user-1", "topic": "elon musk"}


def test_post_reactivates_an_existing_inactive_subscription_instead_of_duplicating(client, fake_prisma):
    existing = FakeRecord(id="sub-1", userId="user-1", topic="elon musk", isActive=False)
    fake_prisma.topicsubscription.find_first.return_value = existing
    _authed_as(client.app, "user-1")

    res = client.post("/subscriptions", json={"topic": "elon musk"})

    assert res.status_code == 200
    fake_prisma.topicsubscription.update.assert_awaited_once_with(
        where={"id": "sub-1"}, data={"isActive": True}
    )
    fake_prisma.topicsubscription.create.assert_not_called()


def test_get_401s_without_auth(client, fake_prisma, monkeypatch):
    monkeypatch.setattr("app.deps.auth.prisma", FakePrisma())
    res = client.get("/subscriptions")
    assert res.status_code == 401


def test_get_scopes_to_the_caller_and_active_only(client, fake_prisma):
    fake_prisma.topicsubscription.find_many.return_value = []
    _authed_as(client.app, "user-1")

    res = client.get("/subscriptions")

    assert res.status_code == 200
    where_clause = fake_prisma.topicsubscription.find_many.call_args.kwargs["where"]
    assert where_clause == {"userId": "user-1", "isActive": True}


def test_delete_deactivates_the_callers_own_subscription(client, fake_prisma):
    existing = FakeRecord(id="sub-1", userId="user-1", topic="elon musk")
    fake_prisma.topicsubscription.find_first.return_value = existing
    _authed_as(client.app, "user-1")

    res = client.delete("/subscriptions?topic=elon musk")

    assert res.status_code == 200
    fake_prisma.topicsubscription.update.assert_awaited_once_with(
        where={"id": "sub-1"}, data={"isActive": False}
    )


def test_delete_is_a_no_op_if_no_matching_subscription(client, fake_prisma):
    fake_prisma.topicsubscription.find_first.return_value = None
    _authed_as(client.app, "user-1")

    res = client.delete("/subscriptions?topic=nonexistent")

    assert res.status_code == 200
    fake_prisma.topicsubscription.update.assert_not_called()
