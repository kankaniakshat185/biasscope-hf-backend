import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Use the REDIS_URL from the environment if available
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
if redis_url.startswith("rediss://") and "ssl_cert_reqs=" not in redis_url:
    join_char = "&" if "?" in redis_url else "?"
    redis_url += f"{join_char}ssl_cert_reqs=CERT_NONE"

# For Upstash compatibility, we often need to ensure rediss:// (SSL) is handled properly
# Celery supports rediss:// natively

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
