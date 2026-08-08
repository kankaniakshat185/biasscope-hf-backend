import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()


def build_redis_url(raw: str) -> str:
    """Appends `ssl_cert_reqs=CERT_NONE` to a `rediss://` URL that doesn't
    already specify it — needed for Upstash-style managed Redis, which
    terminates TLS with a cert chain the default `ssl_cert_reqs=required`
    behavior won't validate. Celery supports rediss:// natively; this is
    only patching the one query param. Extracted for direct testability
    (see tests/test_celery_app.py)."""
    if raw.startswith("rediss://") and "ssl_cert_reqs=" not in raw:
        join_char = "&" if "?" in raw else "?"
        raw += f"{join_char}ssl_cert_reqs=CERT_NONE"
    return raw


# Use the REDIS_URL from the environment if available
redis_url = build_redis_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

celery_app = Celery(
    "biasscope_worker",
    broker=redis_url,
    backend=redis_url,
    include=["app.tasks.snapshot_task"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_redirect_stdouts=False,
    worker_redirect_stdouts_level="INFO",
    broker_transport_options={
        "visibility_timeout": 3600,
        "socket_keepalive": True,
        "socket_timeout": 60,
        "retry_on_timeout": True
    },
    redis_backend_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 60,
        "retry_on_timeout": True
    },
    # Weekly snapshot schedule
    beat_schedule={
        "generate-weekly-snapshots": {
            "task": "app.tasks.snapshot_task.run_weekly_snapshots",
            "schedule": 604800.0, # Every 7 days in seconds
        },
    }
)
