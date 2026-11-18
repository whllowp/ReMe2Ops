"""Cloud evidence ledger: source parsing, conflict counting, idempotent rebuild."""

import json
from pathlib import Path
from unittest.mock import Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.evidence import (
    EVIDENCE_LEDGER_FILE,
    OpsEvidenceStep,
    build_ledger_entries,
    conflict_stats,
    evidence_ledger_path,
    extract_sources,
    load_evidence_ledger,
    parse_wikilinks,
)

PROCEDURE_A = "digest/procedure/disk-cleanup.md"
WIKI_B = "digest/wiki/gateway.md"
MUSCLE_C = "digest/muscle/reboot.md"

BODY_WITH_SOURCES = """# Disk cleanup

Run the cleanup in this order.

## Sources

The order comes from [[daily/2026-11-17/incident.md]] and was corroborated by
[[daily/2026-11-18/followup.md]]; see also [[daily/2026-11-17/incident.md]] again.
"""

BODY_SECOND_SOURCE = """
## Sources

Second opinion: [[daily/2026-11-16/older.md|the older run]] was different.
"""

BODY_NO_SOURCES = "# Gateway\n\nPlain node without provenance.\n"


def _write_node(tmp_path: Path, rel: str, body: str, description: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(body, name=Path(rel).stem, description=description)), encoding="utf-8")


def _append_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"  # cloud aggregate path
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _step(tmp_path: Path):
    return OpsEvidenceStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


# ----- pure functions ---------------------------------------------------------


def test_parse_wikilinks_dedups_and_strips_alias_and_anchor():
    text = "[[a.md]] [[a.md]] [[b.md|alias]] [[c.md#section]] [[#anchor-only]] [[ d.md ]]"
    assert parse_wikilinks(text) == ["a.md", "b.md", "c.md", "d.md"]
    assert parse_wikilinks("") == []


def test_extract_sources_dedup_count_across_sections():
    body = BODY_WITH_SOURCES + BODY_SECOND_SOURCE
    sources = extract_sources(body)
    # the repeated incident link counts once: dedup, not occurrences
    assert sources == [
        "daily/2026-11-16/older.md",
        "daily/2026-11-17/incident.md",
        "daily/2026-11-18/followup.md",
    ]
    assert len(sources) == 3


def test_extract_sources_stops_at_next_heading_and_skips_other_sections():
    body = (
        "## Sources\n\n[[daily/2026-11-17/incident.md]]\n"
        "## Notes\n\nThis mention [[daily/2026-11-10/outside.md]] must not count.\n"
    )
    assert extract_sources(body) == ["daily/2026-11-17/incident.md"]
    assert extract_sources("### Sources\n\n[[a.md]]\n") == []  # only exact h2 counts
    assert extract_sources(BODY_NO_SOURCES) == []


def test_conflict_stats_counts_failed_usage_only():
    records = [
        {"ts": "2026-11-18T01:00:00", "path": PROCEDURE_A, "outcome": "solved"},
        {"ts": "2026-11-18T02:00:00", "path": PROCEDURE_A, "outcome": "failed"},
        {"ts": "2026-11-18T03:00:00", "path": PROCEDURE_A, "outcome": "failed"},
        {"ts": "2026-11-18T04:00:00", "path": PROCEDURE_A, "outcome": "bogus"},  # invalid outcome ignored
        {"ts": "2026-11-18T05:00:00", "path": WIKI_B, "outcome": "solved"},
        "junk-line",
    ]
    stats = conflict_stats(records)
    assert stats[PROCEDURE_A]["failed"] == 2
    assert stats[PROCEDURE_A]["last_used"] == "2026-11-18T03:00:00"  # newest failed ts
    assert WIKI_B not in stats  # solved-only nodes have no conflict
    assert all(entry["failed"] == entry["total"] for entry in stats.values())


def test_build_ledger_entries_shape_and_zero_defaults():
    ledger = build_ledger_entries(
        {PROCEDURE_A: BODY_WITH_SOURCES, WIKI_B: BODY_NO_SOURCES, MUSCLE_C: ""},
        conflict_stats(
            [
                {"ts": "2026-11-18T02:00:00", "path": PROCEDURE_A, "outcome": "failed"},
                {"ts": "2026-11-18T03:00:00", "path": PROCEDURE_A, "outcome": "failed"},
            ]
        ),
    )
    assert ledger[PROCEDURE_A] == {
        "sources": ["daily/2026-11-17/incident.md", "daily/2026-11-18/followup.md"],
        "proof_count": 2,
        "conflict_count": 2,
        "last_conflict_at": "2026-11-18T03:00:00",
    }
    # node without a ## Sources section: proof_count = 0
    assert ledger[WIKI_B] == {"sources": [], "proof_count": 0, "conflict_count": 0, "last_conflict_at": ""}


# ----- step -------------------------------------------------------------------


def _seed_workspace(tmp_path: Path) -> None:
    _write_node(tmp_path, PROCEDURE_A, BODY_WITH_SOURCES, "Disk cleanup SOP")
    _write_node(tmp_path, WIKI_B, BODY_NO_SOURCES, "Gateway notes")
    _append_usage(
        tmp_path,
        [
            {"ts": "2026-11-18T01:00:00", "path": PROCEDURE_A, "outcome": "solved"},
            {"ts": "2026-11-18T02:00:00", "path": PROCEDURE_A, "outcome": "failed"},
            {"ts": "2026-11-18T03:00:00", "path": PROCEDURE_A, "outcome": "failed", "source_device": "edge-01"},
        ],
    )


@pytest.mark.asyncio
async def test_step_joins_sources_and_usage_into_the_ledger(tmp_path):
    _seed_workspace(tmp_path)
    response = await _step(tmp_path)(RuntimeContext())

    ledger = load_evidence_ledger(evidence_ledger_path(tmp_path, "metadata"))
    assert set(ledger) == {PROCEDURE_A, WIKI_B}
    assert ledger[PROCEDURE_A]["proof_count"] == 2  # deduped sources
    assert ledger[PROCEDURE_A]["conflict_count"] == 2  # failed usage only
    assert ledger[PROCEDURE_A]["last_conflict_at"] == "2026-11-18T03:00:00"
    assert ledger[WIKI_B]["proof_count"] == 0 and ledger[WIKI_B]["conflict_count"] == 0
    meta = response.metadata["ops_evidence"]
    assert meta["nodes"] == 2
    assert meta["ledger"] == "metadata/ops_evidence/ledger.json"
    assert meta["proof_total"] == 2 and meta["conflict_total"] == 2
    assert meta["conflict_nodes"] == [PROCEDURE_A]
    assert meta["usage_records"] == 3


@pytest.mark.asyncio
async def test_step_rerun_is_idempotent_same_input_same_json(tmp_path):
    _seed_workspace(tmp_path)
    step = _step(tmp_path)
    await step(RuntimeContext())
    first = evidence_ledger_path(tmp_path, "metadata").read_bytes()

    response2 = await step(RuntimeContext())
    second = evidence_ledger_path(tmp_path, "metadata").read_bytes()

    assert first == second  # byte-identical rebuild
    assert response2.metadata["ops_evidence"]["nodes"] == 2


@pytest.mark.asyncio
async def test_step_writes_only_metadata_shared_response_contract(tmp_path):
    """Same cron window as link_repair: answer/success stay with the chain."""
    _seed_workspace(tmp_path)
    response = await _step(tmp_path)(RuntimeContext())

    assert response.answer == ""  # untouched
    assert response.success is True  # untouched
    assert "ops_evidence" in response.metadata


@pytest.mark.asyncio
async def test_step_tolerates_corrupt_existing_ledger(tmp_path):
    _seed_workspace(tmp_path)
    ledger_path = evidence_ledger_path(tmp_path, "metadata")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("{not valid json!!", encoding="utf-8")

    assert load_evidence_ledger(ledger_path) == {}  # corrupt read tolerated
    response = await _step(tmp_path)(RuntimeContext())  # rebuild overwrites it

    assert response.metadata["ops_evidence"]["nodes"] == 2
    assert load_evidence_ledger(ledger_path)[PROCEDURE_A]["proof_count"] == 2


@pytest.mark.asyncio
async def test_step_without_digest_or_usage_writes_empty_ledger(tmp_path):
    (tmp_path / "digest").mkdir()
    response = await _step(tmp_path)(RuntimeContext())

    meta = response.metadata["ops_evidence"]
    assert meta["nodes"] == 0 and meta["proof_total"] == 0 and meta["conflict_total"] == 0
    assert load_evidence_ledger(evidence_ledger_path(tmp_path, "metadata")) == {}


# ----- config wiring ----------------------------------------------------------


def test_cloud_wiring_job_and_maintenance_slot():
    cloud = _load_config("ops_cloud.yaml")
    job = cloud["jobs"]["ops_evidence"]
    assert job["backend"] == "base"
    assert job["enable_serve"] is True
    assert [step["backend"] for step in job["steps"]] == ["ops_evidence_step"]
    maintenance = [step["backend"] for step in cloud["jobs"]["ops_maintenance_cron"]["steps"]]
    # ledger and reverse index rebuild last: both see the post-repair graph
    assert maintenance.index("ops_evidence_step") > maintenance.index("ops_link_repair_step")
    assert maintenance.index("ops_evidence_step") < maintenance.index("ops_source_index_step")
