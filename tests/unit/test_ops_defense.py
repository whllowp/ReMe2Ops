"""Ops defense scan: pattern pack, redact/block step semantics, config wiring."""

from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import DreamState
from reme.steps.ops.defense import (
    DEFAULT_DEFENSE_MODE,
    DEFENSE_BLOCK,
    DEFENSE_PATTERNS,
    DEFENSE_REDACT,
    OpsDefenseScanStep,
    build_defense_note,
    redact_text,
    redact_unit,
    scan_text,
    scan_unit,
)

SECRET_KEY = "sk-live-abcdef1234567890"
SECRET_GHP = "ghp_AAAAAAAAAAAAAAAAAAAAAAAA"
SECRET_PWD = "hunter2secret"
DIRTY_SUMMARY = f"rotate api_key = \"{SECRET_KEY}\"; deploy token: {SECRET_GHP}; password={SECRET_PWD}"
DIRTY_NAME = "gateway at 192.168.1.10"
CLEAN_UNIT = {"name": "disk-full-runbook", "bucket": "procedure", "summary": "clear the disk, keep logs", "paths": ["daily/2026-11-18/clean.md"]}
DIRTY_UNIT = {"name": DIRTY_NAME, "bucket": "procedure", "summary": DIRTY_SUMMARY, "paths": ["daily/2026-11-18/dirty.md"]}
CLEAN_PATH = "daily/2026-11-18/clean.md"
DIRTY_PATH = "daily/2026-11-18/dirty.md"


def _step(tmp_path, **kwargs):
    return OpsDefenseScanStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        **kwargs,
    )


def _context(units, changed_paths=None, processed_paths=None):
    state = DreamState(
        date="2026-11-18",
        daily_dir="daily",
        units=units,
        changed_paths=changed_paths if changed_paths is not None else sorted({p for u in units for p in u["paths"]}),
        processed_paths=processed_paths,
    )
    return RuntimeContext(dream=state.model_dump())


def _state(context):
    return DreamState.model_validate(context["dream"])


# ----- pattern pack (pure functions) -----------------------------------------


def test_pattern_pack_hits_api_key_token_password_internal_ip():
    assert scan_text(f'api_key = "{SECRET_KEY}"') == ["api_key_assignment", "openai_style_key"]
    assert scan_text(f"deploy token: {SECRET_GHP}") == ["credential_assignment", "github_token"]
    assert scan_text(f"password={SECRET_PWD}") == ["credential_assignment"]
    assert scan_text("gateway at 192.168.1.10 and 172.16.0.9 and 10.0.0.7") == ["internal_ipv4"]
    # public addresses and version-looking numbers are not 内网 IP
    assert scan_text("db at 11.22.33.44, version 10.0.0 released") == []


def test_pattern_pack_is_named_and_compiled():
    names = [pattern.name for pattern in DEFENSE_PATTERNS]
    assert len(names) == len(set(names))  # unique pattern types
    assert all(pattern.regex.groups >= 1 for pattern in DEFENSE_PATTERNS)


def test_redact_text_replaces_only_values_and_reports_applied_patterns():
    text = f'api_key = "{SECRET_KEY}" on 192.168.1.10'
    out, applied = redact_text(text)
    assert SECRET_KEY not in out
    assert "192.168.1.10" not in out
    assert 'api_key = "[REDACTED:api_key_assignment]"' in out
    assert "[REDACTED:internal_ipv4]" in out
    assert applied == ["openai_style_key", "api_key_assignment", "internal_ipv4"]


def test_redact_text_without_hits_is_byte_identical():
    text = "clear the disk, keep logs"
    out, applied = redact_text(text)
    assert out == text
    assert applied == []


def test_scan_unit_and_redact_unit_preserve_other_fields():
    assert scan_unit(DIRTY_UNIT) == ["api_key_assignment", "credential_assignment", "github_token", "internal_ipv4", "openai_style_key"]
    assert scan_unit(CLEAN_UNIT) == []
    scrubbed, applied = redact_unit(DIRTY_UNIT)
    assert applied == scan_unit(DIRTY_UNIT)
    assert scrubbed["bucket"] == "procedure"  # non-scanned fields untouched
    assert scrubbed["paths"] == DIRTY_UNIT["paths"]
    assert SECRET_PWD not in scrubbed["summary"]


def test_default_defense_mode_is_redact():
    assert DEFAULT_DEFENSE_MODE == DEFENSE_REDACT
    assert DEFENSE_BLOCK in ("redact", "block")


def test_build_defense_note_lists_blocked_paths_and_pattern_types_only():
    note = build_defense_note(
        DEFENSE_BLOCK,
        [{"name": DIRTY_NAME, "patterns": ["internal_ipv4"], "paths": [DIRTY_PATH]}],
        [{"name": CLEAN_UNIT["name"], "patterns": ["credential_assignment"], "paths": [CLEAN_PATH]}],
    )
    assert DIRTY_PATH in note
    assert "[[daily/2026-11-18/dirty.md]]" in note
    assert "internal_ipv4" in note
    assert "credential_assignment" in note
    # matched secret values are never recorded in the report
    assert SECRET_KEY not in note
    assert SECRET_PWD not in note
    assert SECRET_GHP not in note


# ----- step: redact mode ------------------------------------------------------


@pytest.mark.asyncio
async def test_redact_mode_rewrites_hits_and_units_continue_downstream(tmp_path):
    step = _step(tmp_path, mode=DEFENSE_REDACT)
    context = _context([DIRTY_UNIT, CLEAN_UNIT])
    response = await step(context)

    state = _state(context)
    assert len(state.units) == 2  # units continue downstream
    assert state.units[0]["name"] == f"gateway at [REDACTED:internal_ipv4]"
    assert SECRET_KEY not in state.units[0]["summary"]
    assert "[REDACTED:api_key_assignment]" in state.units[0]["summary"]
    assert "[REDACTED:internal_ipv4]" in state.units[0]["name"]
    assert state.units[0]["bucket"] == "procedure"
    meta = response.metadata["defense"]
    assert meta["mode"] == "redact"
    assert meta["units_in"] == 2 and meta["units_out"] == 2
    assert meta["redacted"][0]["patterns"] == ["api_key_assignment", "credential_assignment", "github_token", "internal_ipv4", "openai_style_key"]
    assert meta["blocked"] == [] and meta["blocked_paths"] == []
    assert "rejected" not in response.metadata  # nothing was discarded


@pytest.mark.asyncio
async def test_redact_mode_without_hits_keeps_original_unit_objects(tmp_path):
    """No-hit units pass through byte-identical (no rewrite, not even a copy)."""
    step = _step(tmp_path)  # default mode (redact)
    clean_b = {**CLEAN_UNIT, "name": "second-clean", "paths": ["daily/2026-11-18/clean-b.md"]}
    context = _context([CLEAN_UNIT, clean_b])
    await step(context)

    state = _state(context)
    # byte-level passthrough: content equal to the input units, no placeholders
    assert state.units == [CLEAN_UNIT, clean_b]
    assert all("[REDACTED" not in str(unit) for unit in state.units)
    meta = context.response.metadata["defense"]
    assert meta["redacted"] == [] and meta["blocked"] == []
    assert meta["units_in"] == meta["units_out"] == 2


# ----- step: block mode -------------------------------------------------------


@pytest.mark.asyncio
async def test_block_mode_discards_once_with_finish_checkpoint_semantics(tmp_path):
    """Blocked unit leaves units; its path stays checkpointable, never retried.

    dream_finish_step checkpoints a changed path only when it is in
    processed_paths and not in failed_paths (reme/steps/evolve/dream/finish.py:34);
    assert the blocked path satisfies exactly that predicate.
    """
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([DIRTY_UNIT, CLEAN_UNIT], processed_paths=[CLEAN_PATH, DIRTY_PATH])
    response = await step(context)

    state = _state(context)
    assert [unit["name"] for unit in state.units] == [CLEAN_UNIT["name"]]  # blocked unit removed
    assert DIRTY_PATH in state.processed_paths  # source path preserved
    assert DIRTY_PATH not in state.failed_paths  # and never deferred
    # finish.py:34 checkpoint predicate: the discard is consumed once, no retry
    checkpoint = [p for p in state.changed_paths if p in set(state.processed_paths) and p not in set(state.failed_paths)]
    assert checkpoint == [CLEAN_PATH, DIRTY_PATH]
    meta = response.metadata["defense"]
    assert meta["mode"] == "block"
    assert meta["units_in"] == 2 and meta["units_out"] == 1
    assert meta["blocked"] == [{"name": DIRTY_NAME, "patterns": scan_unit(DIRTY_UNIT), "paths": [DIRTY_PATH]}]
    assert meta["blocked_paths"] == [DIRTY_PATH]
    assert response.metadata["rejected"] == [f"defense blocked unit {DIRTY_NAME!r}: " + "+".join(scan_unit(DIRTY_UNIT))]


@pytest.mark.asyncio
async def test_block_mode_defends_paths_missing_from_processed_paths(tmp_path):
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([DIRTY_UNIT], processed_paths=[])  # extract normally covers this
    await step(context)

    state = _state(context)
    assert state.units == []
    assert state.processed_paths == [DIRTY_PATH]  # defensively re-added
    assert state.failed_paths == []


@pytest.mark.asyncio
async def test_block_mode_writes_daily_report_with_provenance(tmp_path):
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([DIRTY_UNIT], processed_paths=[DIRTY_PATH])
    writes: list[tuple[str, dict]] = []

    async def fake_run_job(name, /, **kwargs):
        writes.append((name, kwargs))
        if name == "write":
            target = tmp_path / kwargs["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = kwargs.get("metadata") or {}
            target.write_text(frontmatter.dumps(frontmatter.Post(kwargs["content"], **metadata)), encoding="utf-8")
        return Mock(success=True, answer="ok")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(context)

    note = response.metadata["defense"]["note"]
    assert note and note.startswith("daily/2026-11-18/ops-defense-")
    assert [name for name, _ in writes] == ["write"]  # only the report write
    kwargs = writes[0][1]
    assert kwargs["metadata"]["source"] == "ops_defense_scan"  # frontmatter provenance
    assert kwargs["metadata"]["mode"] == "block"
    assert kwargs["metadata"]["blocked_paths"] == [DIRTY_PATH]
    on_disk = frontmatter.loads((tmp_path / note).read_text(encoding="utf-8"))
    assert on_disk.metadata["source"] == "ops_defense_scan"
    assert DIRTY_PATH in on_disk.content
    assert "internal_ipv4" in on_disk.content
    assert SECRET_KEY not in on_disk.content and SECRET_PWD not in on_disk.content


@pytest.mark.asyncio
async def test_block_mode_report_write_failure_does_not_break_ingest(tmp_path):
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([DIRTY_UNIT], processed_paths=[DIRTY_PATH])
    step.run_job = AsyncMock(side_effect=RuntimeError("no write job"))
    response = await step(context)

    meta = response.metadata["defense"]
    assert meta["note"] == ""
    assert meta["report_error"].startswith("RuntimeError")
    assert meta["units_out"] == 0  # the discard itself still happened
    assert response.success is not False  # shared response untouched by defense


@pytest.mark.asyncio
async def test_block_mode_appends_to_existing_cloud_rejected_channel(tmp_path):
    """cloud_ingest's seed step writes metadata['rejected']; defense extends it."""
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([DIRTY_UNIT], processed_paths=[DIRTY_PATH])
    context.response.metadata["rejected"] = ["units[0] missing name/summary/paths"]
    await step(context)

    rejected = context.response.metadata["rejected"]
    assert rejected[0] == "units[0] missing name/summary/paths"  # seed reasons preserved
    assert len(rejected) == 2 and rejected[1].startswith("defense blocked unit")


# ----- passthrough ------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_units_is_a_passthrough(tmp_path):
    step = _step(tmp_path, mode=DEFENSE_BLOCK)
    context = _context([])
    response = await step(context)

    meta = response.metadata["defense"]
    assert meta["units_in"] == 0 and meta["units_out"] == 0
    assert meta["redacted"] == [] and meta["blocked"] == [] and meta["blocked_paths"] == []
    assert meta["note"] == "" and meta["report_error"] == ""
    assert "rejected" not in response.metadata


@pytest.mark.asyncio
async def test_invalid_mode_falls_back_to_redact(tmp_path):
    step = _step(tmp_path, mode="nonsense")
    context = _context([DIRTY_UNIT])
    await step(context)

    meta = context.response.metadata["defense"]
    assert meta["mode"] == "redact"
    assert meta["blocked"] == [] and meta["units_out"] == 1


# ----- config wiring ----------------------------------------------------------


def test_edge_dream_push_chain_carries_the_defense_gate():
    edge = _load_config("ops_edge.yaml")
    expected = ["dream_extract_step", "ops_defense_scan_step", "ops_push_step", "dream_finish_step"]
    base = edge["jobs"]["ops_dream_push"]
    assert [step["backend"] for step in base["steps"]] == expected
    assert base["steps"][1]["mode"] == "redact"  # edge outbound gate: scrub, keep memory
    cron = edge["jobs"]["ops_dream_push_cron"]
    assert [step["backend"] for step in cron["steps"]] == expected  # anchor survives edits
    assert cron["steps"][1]["mode"] == "redact"


def test_cloud_ingest_chain_carries_the_second_gate():
    cloud = _load_config("ops_cloud.yaml")
    job = cloud["jobs"]["cloud_ingest"]
    backends = [step["backend"] for step in job["steps"]]
    assert backends == ["ops_ingest_seed_step", "ops_defense_scan_step", "dream_integrate_step", "auto_tag_step"]
    assert job["steps"][1]["mode"] == "block"  # cloud inbound gate: one-shot discard
