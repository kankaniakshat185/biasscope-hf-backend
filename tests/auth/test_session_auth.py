"""app/deps/auth.py — the session-cookie auth dependency that IS the fix
for the audit's top two findings (S1: no auth on any route; S2: IDOR via
client-supplied userId). This is the highest-value place in the whole
backend to have real regression tests: a silent regression here quietly
reopens both.
"""

from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.deps import auth as auth_module
from tests.fakes import FakePrisma, fake_request, fake_session


@pytest.fixture
def fake_prisma(monkeypatch):
    prisma = FakePrisma()
    monkeypatch.setattr(auth_module, "prisma", prisma)
    return prisma


# ── get_current_user_id ──────────────────────────────────────────────

async def test_rejects_request_with_no_cookie_or_header(fake_prisma):
    request = fake_request()
    with pytest.raises(HTTPException) as exc_info:
        await auth_module.get_current_user_id(request)
    assert exc_info.value.status_code == 401


async def test_accepts_valid_session_cookie(fake_prisma):
    fake_prisma.session.find_first.return_value = fake_session(userId="user-42")
    request = fake_request(cookies={"better-auth.session_token": "tok_abc123"})

    user_id = await auth_module.get_current_user_id(request)

    assert user_id == "user-42"


async def test_accepts_secure_prefixed_cookie_name(fake_prisma):
    fake_prisma.session.find_first.return_value = fake_session(userId="user-42")
    request = fake_request(cookies={"__Secure-better-auth.session_token": "tok_abc123"})

    user_id = await auth_module.get_current_user_id(request)

    assert user_id == "user-42"


async def test_accepts_bearer_token_header(fake_prisma):
    fake_prisma.session.find_first.return_value = fake_session(userId="user-42")
    request = fake_request(headers={"authorization": "Bearer tok_abc123"})

    user_id = await auth_module.get_current_user_id(request)

    assert user_id == "user-42"


async def test_tries_signed_cookie_prefix_before_the_dot(fake_prisma):
    # Better Auth sometimes signs cookies as "<token>.<signature>" — the
    # dependency should also try the part before the dot as a candidate.
    fake_prisma.session.find_first.return_value = fake_session(userId="user-42")
    request = fake_request(cookies={"better-auth.session_token": "tok_abc123.signature_junk"})

    user_id = await auth_module.get_current_user_id(request)

    assert user_id == "user-42"
    queried_tokens = fake_prisma.session.find_first.call_args.kwargs["where"]["token"]["in"]
    assert "tok_abc123" in queried_tokens
    assert "tok_abc123.signature_junk" in queried_tokens


async def test_rejects_when_no_session_row_matches(fake_prisma):
    fake_prisma.session.find_first.return_value = None
    request = fake_request(cookies={"better-auth.session_token": "not-a-real-token"})

    with pytest.raises(HTTPException) as exc_info:
        await auth_module.get_current_user_id(request)
    assert exc_info.value.status_code == 401


async def test_rejects_expired_session():
    prisma = FakePrisma()
    prisma.session.find_first.return_value = fake_session(expires_in_future=False)
    import app.deps.auth as m
    m.prisma = prisma
    request = fake_request(cookies={"better-auth.session_token": "tok_abc123"})

    with pytest.raises(HTTPException) as exc_info:
        await m.get_current_user_id(request)
    assert exc_info.value.status_code == 401


async def test_accepts_naive_datetime_expiry_as_utc(fake_prisma):
    # Prisma may return naive datetimes depending on driver config — the
    # expiry check must not crash comparing naive vs aware datetimes.
    naive_future = datetime.now() + timedelta(hours=1)
    naive_future = naive_future.replace(tzinfo=None)
    fake_prisma.session.find_first.return_value = fake_session()
    fake_prisma.session.find_first.return_value.expiresAt = naive_future
    request = fake_request(cookies={"better-auth.session_token": "tok_abc123"})

    user_id = await auth_module.get_current_user_id(request)
    assert user_id == "user-1"


# ── get_optional_user_id ─────────────────────────────────────────────

async def test_optional_returns_none_when_anonymous(fake_prisma):
    request = fake_request()
    assert await auth_module.get_optional_user_id(request) is None


async def test_optional_returns_user_id_when_logged_in(fake_prisma):
    fake_prisma.session.find_first.return_value = fake_session(userId="user-99")
    request = fake_request(cookies={"better-auth.session_token": "tok_abc123"})

    assert await auth_module.get_optional_user_id(request) == "user-99"


async def test_optional_never_raises_for_invalid_session(fake_prisma):
    fake_prisma.session.find_first.return_value = None
    request = fake_request(cookies={"better-auth.session_token": "garbage"})

    # Must degrade to "anonymous", not blow up — /search allows anonymous callers.
    assert await auth_module.get_optional_user_id(request) is None


# ── debug route gating ───────────────────────────────────────────────

def test_debug_routes_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ROUTES", raising=False)
    assert auth_module.debug_routes_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "YES"])
def test_debug_routes_enabled_values(monkeypatch, value):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", value)
    assert auth_module.debug_routes_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "garbage"])
def test_debug_routes_disabled_values(monkeypatch, value):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", value)
    assert auth_module.debug_routes_enabled() is False


async def test_require_debug_enabled_404s_when_off(monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ROUTES", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await auth_module.require_debug_enabled()
    # 404, not 403 — the route shouldn't even reveal it exists when off.
    assert exc_info.value.status_code == 404


async def test_require_debug_enabled_passes_when_on(monkeypatch):
    monkeypatch.setenv("ENABLE_DEBUG_ROUTES", "1")
    await auth_module.require_debug_enabled()  # must not raise
