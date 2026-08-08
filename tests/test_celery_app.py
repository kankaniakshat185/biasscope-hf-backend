"""app/celery_app.py — mostly static config, but the rediss:// SSL-param
patch (needed for Upstash-style managed Redis) is real logic worth
pinning down. Importing this module is safe: constructing a Celery() app
doesn't eagerly connect to the broker."""

from app.celery_app import build_redis_url, celery_app


def test_appends_ssl_cert_reqs_to_a_bare_rediss_url():
    assert build_redis_url("rediss://user:pass@host:6379/0") == (
        "rediss://user:pass@host:6379/0?ssl_cert_reqs=CERT_NONE"
    )


def test_appends_with_ampersand_when_a_query_string_already_exists():
    assert build_redis_url("rediss://host:6379/0?foo=bar") == (
        "rediss://host:6379/0?foo=bar&ssl_cert_reqs=CERT_NONE"
    )


def test_does_not_duplicate_an_already_present_ssl_cert_reqs():
    url = "rediss://host:6379/0?ssl_cert_reqs=required"
    assert build_redis_url(url) == url


def test_plain_redis_url_is_left_untouched():
    url = "redis://localhost:6379/0"
    assert build_redis_url(url) == url


def test_weekly_snapshot_beat_schedule_is_registered():
    schedule = celery_app.conf.beat_schedule["generate-weekly-snapshots"]
    assert schedule["task"] == "app.tasks.snapshot_task.run_weekly_snapshots"
    assert schedule["schedule"] == 604800.0  # 7 days, in seconds


def test_snapshot_task_module_is_registered_for_the_worker_to_discover():
    assert "app.tasks.snapshot_task" in celery_app.conf.include
