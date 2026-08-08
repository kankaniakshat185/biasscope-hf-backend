"""
app/main.py — app wiring only (see its own docstring). Tests here call
`startup()`/`shutdown()` directly as plain coroutines with `app.main.prisma`
monkeypatched to a fake FIRST, rather than exercising them through
TestClient's ASGI lifespan — this repo's local .env has a real production
DATABASE_URL in it, and a test must never risk opening that connection
by accident. See tests/routers/conftest.py for the same concern on the
router side.
"""


from app import main as main_module
from tests.fakes import FakePrisma


def test_parse_allowed_origins_defaults_to_localhost_and_the_known_vercel_url():
    # Regression: this used to assert "biasscope-app-frontend.vercel.app",
    # which doesn't match the real deployed site at all (confirmed against
    # the actual production URL's address bar) — every credentialed
    # request from production was silently CORS-rejected, and this test
    # was pinning the wrong value the whole time.
    origins = main_module.parse_allowed_origins(None)
    assert "http://localhost:3000" in origins
    assert "https://biasscope-app.vercel.app" in origins


def test_parse_allowed_origins_splits_and_trims_a_custom_value():
    origins = main_module.parse_allowed_origins(" https://a.com , https://b.com ")
    assert origins == ["https://a.com", "https://b.com"]


def test_parse_allowed_origins_drops_empty_entries():
    origins = main_module.parse_allowed_origins("https://a.com,,https://b.com,")
    assert origins == ["https://a.com", "https://b.com"]


def test_parse_allowed_origins_empty_string_falls_back_to_default():
    # os.environ.get("ALLOWED_ORIGINS") returns "" if someone sets the var
    # to an empty string (distinct from it being unset) — must not produce
    # an accidental allow-nothing (or worse, allow-empty-string-origin) policy.
    origins = main_module.parse_allowed_origins("")
    assert "http://localhost:3000" in origins


def test_cors_never_pairs_wildcard_with_credentials():
    # The actual regression this whole thing exists to prevent (S1).
    for middleware in main_module.app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            kwargs = middleware.kwargs
            if kwargs.get("allow_credentials"):
                assert kwargs.get("allow_origins") != ["*"]


def test_all_routers_are_mounted():
    paths = {route.path for route in main_module.app.routes}
    for expected in ["/", "/search", "/history", "/subscriptions", "/chat-with-article", "/debug/status"]:
        assert expected in paths, f"{expected} is not mounted"


def test_read_root_reports_ok():
    assert main_module.read_root() == {"status": "ok", "service": "Biascope Backend"}


async def test_startup_connects_prisma(monkeypatch):
    fake_prisma = FakePrisma()
    monkeypatch.setattr(main_module, "prisma", fake_prisma)
    await main_module.startup()
    fake_prisma.connect.assert_awaited_once()


async def test_shutdown_disconnects_prisma(monkeypatch):
    fake_prisma = FakePrisma()
    monkeypatch.setattr(main_module, "prisma", fake_prisma)
    await main_module.shutdown()
    fake_prisma.disconnect.assert_awaited_once()
