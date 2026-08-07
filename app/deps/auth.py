"""
Session-based authentication for the FastAPI backend.

Better Auth (the frontend's auth library) already maintains `session` and
`user` tables in the same Postgres database this backend talks to. Rather
than inventing a second auth system, we verify the session cookie/token the
frontend already has by looking it up directly in that shared table.

This is intentionally the *only* place identity is resolved. Every route
that touches user-owned data must depend on `get_current_user_id` (require
login) or `get_optional_user_id` (login optional) instead of trusting a
`userId` value from the request body/query string — a client can lie about
a body param, it cannot fabricate a valid row in the `session` table.

NOTE: Better Auth's default cookie name is `better-auth.session_token`
(or the `__Secure-` prefixed variant when secure cookies are enabled), and
the cookie value is sometimes `<token>.<signature>`. If your installed
version differs, requests will 401 across the board — inspect
`request.cookies` against what the browser actually sends and adjust
SESSION_COOKIE_NAMES below.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, Request, status

from ..db import prisma

SESSION_COOKIE_NAMES = (
    "__Secure-better-auth.session_token",
    "better-auth.session_token",
)


def _candidate_tokens(request: Request) -> List[str]:
    """Collect every plausible session-token value from the request.

    Checks, in order: an `Authorization: Bearer <token>` header (useful for
    non-browser callers), then each known cookie name — both the raw cookie
    value and, if it looks like `<token>.<signature>`, the part before the
    first dot (Better Auth signs cookies by default in some configurations).
    """
    candidates: List[str] = []

    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        candidates.append(auth_header[7:].strip())

    for name in SESSION_COOKIE_NAMES:
        raw = request.cookies.get(name)
        if not raw:
            continue
        candidates.append(raw)
        if "." in raw:
            candidates.append(raw.split(".", 1)[0])

    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _is_expired(expires_at: Optional[datetime]) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < datetime.now(timezone.utc)


async def _lookup_session(request: Request):
    tokens = _candidate_tokens(request)
    if not tokens:
        return None

    session = await prisma.session.find_first(where={"token": {"in": tokens}})
    if not session or _is_expired(session.expiresAt):
        return None
    return session


async def get_current_user_id(request: Request) -> str:
    """Require a valid session. Raises 401 if there isn't one."""
    session = await _lookup_session(request)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return session.userId


async def get_optional_user_id(request: Request) -> Optional[str]:
    """Resolve the caller's user id if logged in; None if anonymous.

    For routes (like /search) that are allowed to run anonymously but must
    never trust a client-supplied user id for the ones who ARE logged in.
    """
    session = await _lookup_session(request)
    return session.userId if session else None


def debug_routes_enabled() -> bool:
    return os.environ.get("ENABLE_DEBUG_ROUTES", "").strip().lower() in ("1", "true", "yes")


async def require_debug_enabled() -> None:
    """Blocks the entire /debug router unless explicitly enabled.

    Returns 404 rather than 403 so the router's existence isn't
    fingerprintable when it's off (the default, and what production should
    run with).
    """
    if not debug_routes_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
