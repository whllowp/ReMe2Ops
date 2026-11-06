"""Ops usage records: storage roundtrip and aggregation for feedback/promotion."""

from reme.steps.ops.usage import (
    aggregate_usage,
    append_usage_record,
    normalize_outcome,
    read_usage_records,
    usage_file_path,
)


def test_usage_file_path_splits_edge_and_cloud_namespaces(tmp_path):
    edge = usage_file_path(tmp_path, "metadata")
    cloud = usage_file_path(tmp_path, "metadata", cloud=True)
    assert edge == tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    assert cloud == tmp_path / "metadata" / "ops_usage" / "usage.jsonl"


def test_append_and_read_usage_records_roundtrip(tmp_path):
    path = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    append_usage_record(path, {"ts": "2026-11-06T10:00:00", "path": "digest/procedure/a.md", "outcome": "solved"})
    append_usage_record(path, {"ts": "2026-11-06T11:00:00", "path": "digest/procedure/a.md", "outcome": "failed"})
    records = read_usage_records(path)
    assert [r["outcome"] for r in records] == ["solved", "failed"]


def test_read_usage_records_tolerates_missing_and_corrupted(tmp_path):
    missing = read_usage_records(tmp_path / "nope.jsonl")
    assert missing == []
    corrupted = tmp_path / "bad.jsonl"
    corrupted.write_text(
        '{"path": "digest/procedure/a.md", "outcome": "solved"}\n'
        "not-json-at-all\n"
        "\n"
        '{"path": "digest/procedure/b.md", "outcome": "failed"}\n',
        encoding="utf-8",
    )
    records = read_usage_records(corrupted)
    assert [r["path"] for r in records] == ["digest/procedure/a.md", "digest/procedure/b.md"]


def test_normalize_outcome():
    assert normalize_outcome("Solved") == "solved"
    assert normalize_outcome(" failed ") == "failed"
    assert normalize_outcome("maybe") is None
    assert normalize_outcome("") is None
    assert normalize_outcome(None) is None


def test_aggregate_usage_counts_and_rates():
    records = [
        {"path": "digest/procedure/a.md", "outcome": "solved", "ts": "2026-11-01T08:00:00"},
        {"path": "digest/procedure/a.md", "outcome": "solved", "ts": "2026-11-02T08:00:00"},
        {"path": "digest/procedure/a.md", "outcome": "failed", "ts": "2026-11-03T08:00:00"},
        {"path": "digest/muscle/m.md", "outcome": "solved", "ts": "2026-11-03T09:00:00"},
    ]
    stats = aggregate_usage(records)
    assert stats["digest/procedure/a.md"] == {
        "solved": 2,
        "failed": 1,
        "total": 3,
        "success_rate": 0.6667,
        "last_used": "2026-11-03T08:00:00",
    }
    assert stats["digest/muscle/m.md"]["solved"] == 1


def test_aggregate_usage_ignores_invalid_records():
    stats = aggregate_usage(
        [
            {"path": "", "outcome": "solved"},
            {"path": "digest/procedure/a.md", "outcome": "maybe"},
            "not-a-dict",
            {},
        ],
    )
    assert stats == {}
