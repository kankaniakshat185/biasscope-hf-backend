from app.tasks.snapshot_task import run_weekly_snapshots
import sys

if __name__ == "__main__":
    result = run_weekly_snapshots.delay()
    print(f"Task sent to Celery! Task ID: {result.id}")
