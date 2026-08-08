"""app/utils/create_demo_snapshot.py's build_snapshot_json — regression
test for a real bug found via mypy while adding type-checking to this
repo (see AUDIT_TASKS.md's Tooling section): passing a pre-serialized
JSON string for a `Json`-typed column instead of wrapping it in Json(...)
made the generated client double-encode it, so DemoSnapshot.data held a
JSON string instead of a JSON object."""

from datetime import datetime

from app.prisma_client import Json
from app.utils.create_demo_snapshot import build_snapshot_json


def test_returns_a_json_wrapper_not_a_plain_string():
    result = build_snapshot_json({"topic": "elon musk"})
    assert isinstance(result, Json)
    assert not isinstance(result, str) or isinstance(result, Json)  # Json subclasses str at runtime; identity matters


def test_wrapped_data_round_trips_to_the_original_structure():
    demo_data = {"id": "demo-1", "topic": "elon musk", "search": {"query": "elon musk"}, "intelligence": {"events": []}}
    result = build_snapshot_json(demo_data)
    assert result.data == demo_data


def test_normalizes_embedded_datetime_objects_instead_of_crashing():
    # Prisma's own Json serializer does a plain json.dumps with no
    # `default=` — this must pre-normalize datetimes into strings, or the
    # eventual write would blow up with "Object of type datetime is not
    # JSON serializable".
    demo_data = {"createdAt": datetime(2026, 1, 1, 12, 0, 0)}
    result = build_snapshot_json(demo_data)
    assert isinstance(result.data["createdAt"], str)


def test_nested_datetimes_in_evidence_lists_are_also_normalized():
    demo_data = {
        "intelligence": {
            "events": [{"evidence": [{"sentence": "X happened.", "publishedAt": datetime(2026, 1, 1)}]}]
        }
    }
    result = build_snapshot_json(demo_data)
    published_at = result.data["intelligence"]["events"][0]["evidence"][0]["publishedAt"]
    assert isinstance(published_at, str)
