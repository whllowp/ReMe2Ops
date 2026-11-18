"""Cloud failure notes: write-back resolution, note merging, no-file-creation rule."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.failure_notes import (
    FAILURE_NOTES_CAP,
    OpsFailureNotesStep,
    apply_failure_notes,
    collect_failures,
    merge_failure_notes,
    note_date,
    note_query,
    parse_failure_notes,
    promoted_canonicals,
    render_failure_section,
)

PROCEDURE = "digest/procedure/disk-cleanup.md"
MUSCLE = "digest/muscle/disk-cleanup.md"

CANONICAL_BODY = """# Disk cleanup

Run the cleanup in this order.

## Sources

From [[daily/2026-11-17/incident.md]].
"""


def _record(ts, path, outcome, query=None, session=None):
    record = {"ts": ts, "path": path, "outcome": outcome}
    if query is not None:
        record["query"] = query
    if session is not None:
        record["session_id"] = session
    return record


def _write_sop(tmp_path: Path, rel: str, body: str, metadata: dict | None = None) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"name": Path(rel).stem, "description": "Disk cleanup SOP"}
    meta.update(metadata or {})
    path.write_text(frontmatter.dumps(frontmatter.Post(body, **meta)), encoding="utf-8")


def _append_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _step(tmp_path: Path, **kwargs):
    return OpsFailureNotesStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        **kwargs,
    )


async def _fake_write_job(step, tmp_path: Path):
    """run_job mock that actually applies write jobs so files can be asserted."""

    async def fake_run_job(name, /, **kwargs):
        if name == "write":
            target = tmp_path / kwargs["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(kwargs.get("metadata") or {})
            meta.setdefault("name", kwargs["name"])
            meta.setdefault("description", kwargs["description"])
            body_text = frontmatter.dumps(frontmatter.Post(kwargs["content"], **meta))
            # mirror the real WriteStep: the file always ends with a newline
            if not body_text.endswith("\n"):
                body_text += "\n"
            target.write_text(body_text, encoding="utf-8")
            return Mock(success=True, answer="ok")
        return Mock(success=True, answer="ok")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    return step


# ----- pure functions ---------------------------------------------------------


def test_note_date_and_query_fallbacks():
    assert note_date("2026-11-18T01:00:00") == "2026-11-18"
    assert note_date("") == "unknown"
    assert note_date("short") == "unknown"
    assert note_query({"query": " disk full "}) == "disk full"
    assert note_query({"session_id": "s9"}) == "session:s9"
    assert note_query({}) == "unrecorded query"


def test_collect_failures_keeps_failed_only_and_dedupes_per_query():
    failures = collect_failures(
        [
            _record("2026-11-18T01:00:00", PROCEDURE, "solved", query="disk full"),
            _record("2026-11-18T02:00:00", PROCEDURE, "failed", query="disk full"),
            _record("2026-11-18T03:00:00", PROCEDURE, "failed", query="disk full"),  # same query: newest ts wins
            _record("2026-11-18T04:00:00", PROCEDURE, "failed", query="wrong flag"),  # distinct query
            _record("2026-11-18T05:00:00", "", "failed", query="no path"),  # ignored
            "junk",
        ],
    )
    assert failures[PROCEDURE] == [("2026-11-18", "disk full"), ("2026-11-18", "wrong flag")]  # query asc within a date


def test_merge_failure_notes_dedupes_sorts_and_caps():
    existing = [("2026-11-10", "old lesson")]
    incoming = [
        ("2026-11-18", "disk full"),
        ("2026-11-17", "disk full"),  # same query, older date -> keep 11-18
        ("2026-11-19", "wrong flag"),
    ]
    merged = merge_failure_notes(existing, incoming)
    assert merged[0] == ("2026-11-19", "wrong flag")  # newest first
    assert ("2026-11-18", "disk full") in merged and ("2026-11-17", "disk full") not in merged
    assert ("2026-11-10", "old lesson") in merged
    assert len(merge_failure_notes([], [(f"2026-11-{10 + i:02d}", f"q{i}") for i in range(8)])) == FAILURE_NOTES_CAP
    oldest = merge_failure_notes([], [(f"2026-11-{10 + i:02d}", f"q{i}") for i in range(8)])[-1]
    assert oldest == ("2026-11-13", "q3")  # the five newest kept: 2026-11-13..17


def test_parse_and_apply_roundtrip_and_idempotence():
    body = CANONICAL_BODY
    notes = [("2026-11-18", "disk full"), ("2026-11-17", "wrong flag")]
    once = apply_failure_notes(body, notes)
    assert "## Failure Notes" in once and "- 2026-11-18 — disk full" in once
    assert parse_failure_notes(once) == notes
    twice = apply_failure_notes(once, notes)
    assert twice == once  # idempotent
    assert parse_failure_notes(body) == []  # untouched original
    # section replaced in place, ## Sources section preserved intact
    assert "## Sources" in twice and twice.index("## Sources") < twice.index("## Failure Notes")


def test_render_failure_section_is_deterministic():
    assert render_failure_section([("2026-11-18", "disk full")]) == "## Failure Notes\n\n- 2026-11-18 — disk full"


def test_promoted_canonicals_maps_alias_to_muscle_path():
    canonicals = promoted_canonicals(
        {
            MUSCLE: {"promoted_from": PROCEDURE},
            "digest/muscle/other.md": {},
        },
    )
    assert canonicals == {PROCEDURE: MUSCLE}
    assert promoted_canonicals({"digest/muscle/other.md": {}}) == {}


# ----- step -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_path_is_written_directly(tmp_path):
    _write_sop(tmp_path, PROCEDURE, CANONICAL_BODY)
    _append_usage(tmp_path, [_record("2026-11-18T01:00:00", PROCEDURE, "failed", query="disk full")])
    step = await _fake_write_job(_step(tmp_path), tmp_path)
    response = await step(RuntimeContext())

    post = frontmatter.loads((tmp_path / PROCEDURE).read_text(encoding="utf-8"))
    assert "## Failure Notes" in post.content and "- 2026-11-18 — disk full" in post.content
    assert post.metadata["failure_count"] == 1
    assert "## Sources" in post.content  # rest of the body and frontmatter preserved
    assert "promoted_from" not in post.metadata  # no alias invented on the direct path
    meta = response.metadata["failure_notes"]
    assert meta["updated"] == [{"usage_path": PROCEDURE, "path": PROCEDURE, "failure_count": 1, "notes": ["2026-11-18 — disk full"]}]
    assert meta["skipped"] == []


@pytest.mark.asyncio
async def test_promoted_path_resolves_to_muscle_canonical_and_never_recreates(tmp_path):
    """Old path gone + muscle canonical carries promoted_from -> notes land there.

    The double-copy chain (recreate procedure/x.md -> promotion scan re-scores
    the prefix -> move(overwrite=True) clobbers the canonical) is cut by never
    creating the old path again.
    """
    _write_sop(tmp_path, MUSCLE, CANONICAL_BODY, metadata={"promoted_from": PROCEDURE, "memory_type": "muscle"})
    _append_usage(tmp_path, [_record("2026-11-18T01:00:00", PROCEDURE, "failed", query="disk full")])
    step = await _fake_write_job(_step(tmp_path), tmp_path)
    response = await step(RuntimeContext())

    assert not (tmp_path / PROCEDURE).exists()  # old path NOT rebuilt
    post = frontmatter.loads((tmp_path / MUSCLE).read_text(encoding="utf-8"))
    assert "- 2026-11-18 — disk full" in post.content
    assert post.metadata["failure_count"] == 1
    assert post.metadata["promoted_from"] == PROCEDURE  # alias preserved through the rewrite
    assert post.metadata["memory_type"] == "muscle"
    meta = response.metadata["failure_notes"]
    assert meta["updated"][0]["path"] == MUSCLE
    assert meta["skipped"] == []


@pytest.mark.asyncio
async def test_unresolvable_path_is_skipped_without_writing(tmp_path):
    _append_usage(tmp_path, [_record("2026-11-18T01:00:00", "digest/procedure/ghost.md", "failed", query="disk full")])
    step = await _fake_write_job(_step(tmp_path), tmp_path)
    response = await step(RuntimeContext())

    assert not (tmp_path / "digest" / "procedure" / "ghost.md").exists()
    meta = response.metadata["failure_notes"]
    assert meta["updated"] == []
    assert meta["skipped"] == [{"path": "digest/procedure/ghost.md", "reason": "no existing file or promoted canonical"}]


@pytest.mark.asyncio
async def test_same_query_recorded_once_and_cap_evicts_oldest(tmp_path):
    _write_sop(tmp_path, PROCEDURE, CANONICAL_BODY)
    _append_usage(
        tmp_path,
        [
            _record("2026-11-18T01:00:00", PROCEDURE, "failed", query="disk full"),
            _record("2026-11-18T02:00:00", PROCEDURE, "failed", query="disk full"),  # duplicate query
            _record("2026-11-19T03:00:00", PROCEDURE, "failed", query="wrong flag"),  # newer date
        ],
    )
    step = await _fake_write_job(_step(tmp_path, max_failure_notes=1), tmp_path)
    await step(RuntimeContext())

    post = frontmatter.loads((tmp_path / PROCEDURE).read_text(encoding="utf-8"))
    notes = parse_failure_notes(post.content)
    assert notes == [("2026-11-19", "wrong flag")]  # newest kept, cap=1 evicted disk full
    assert post.metadata["failure_count"] == 1


@pytest.mark.asyncio
async def test_solved_only_usage_never_touches_the_file(tmp_path):
    _write_sop(tmp_path, PROCEDURE, CANONICAL_BODY)
    before = (tmp_path / PROCEDURE).read_bytes()
    _append_usage(tmp_path, [_record("2026-11-18T01:00:00", PROCEDURE, "solved", query="disk full")])
    step = await _fake_write_job(_step(tmp_path), tmp_path)
    response = await step(RuntimeContext())

    assert (tmp_path / PROCEDURE).read_bytes() == before  # byte-identical: no note, no failure_count
    assert response.metadata["failure_notes"]["updated"] == []
    assert response.metadata["failure_notes"]["failures_seen"] == 0


@pytest.mark.asyncio
async def test_rerun_after_write_is_a_noop_no_mtime_churn(tmp_path):
    _write_sop(tmp_path, PROCEDURE, CANONICAL_BODY)
    _append_usage(tmp_path, [_record("2026-11-18T01:00:00", PROCEDURE, "failed", query="disk full")])
    step = await _fake_write_job(_step(tmp_path), tmp_path)
    await step(RuntimeContext())
    after_first = (tmp_path / PROCEDURE).read_bytes()

    write_calls = []
    original = step.run_job  # the fake from the first run

    async def counting_run_job(name, /, **kwargs):
        write_calls.append(name)
        return await original(name, **kwargs)

    step.run_job = AsyncMock(side_effect=counting_run_job)
    response2 = await step(RuntimeContext())

    assert (tmp_path / PROCEDURE).read_bytes() == after_first  # same content
    assert "write" not in write_calls  # merged notes already present -> no write
    assert response2.metadata["failure_notes"]["updated"] == []


# ----- config wiring ----------------------------------------------------------


def test_cloud_wiring_job_and_maintenance_slot():
    cloud = _load_config("ops_cloud.yaml")
    job = cloud["jobs"]["ops_failure_notes"]
    assert job["backend"] == "base"
    assert job["enable_serve"] is True
    assert job["steps"][0]["backend"] == "ops_failure_notes_step"
    assert job["steps"][0]["max_failure_notes"] == 5
    maintenance = [step["backend"] for step in cloud["jobs"]["ops_maintenance_cron"]["steps"]]
    assert maintenance.index("ops_failure_notes_step") > maintenance.index("ops_link_repair_step")
    assert maintenance.index("ops_failure_notes_step") < maintenance.index("ops_evidence_step")
    assert maintenance.index("ops_failure_notes_step") < maintenance.index("ops_source_index_step")
