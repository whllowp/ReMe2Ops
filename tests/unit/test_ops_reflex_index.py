"""Bounded reflex index: entries, deterministic render, shared-Response contract."""

import json
from pathlib import Path
from unittest.mock import Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.dream_audit import audit_status_path, body_sha256
from reme.steps.ops.reflex_index import (
    INDEX_MAX_BYTES,
    INDEX_MAX_LINES,
    OpsReflexIndexStep,
    index_entries,
    recent_daily_pointers,
    render_memory_index,
)


def _write_mirror(tmp_path: Path, rel: str, body: str, metadata: dict) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(body, **metadata)), encoding="utf-8")


def test_index_entries_hash_gated_status():
    mirrors = {
        "digest/muscle/a.md": {"description": "Disk cleanup", "scope": "general", "body": "body-a"},
        "digest/muscle/b.md": {"description": "Log rotate", "scope": "device:edge-1", "body": "body-b"},
        "digest/muscle/c.md": {"description": "", "scope": "general", "body": "body-c"},
    }
    ledger = {
        "digest/muscle/a.md": {"status": "confirmed", "body_sha256": body_sha256("body-a")},
        "digest/muscle/b.md": {"status": "conflict", "body_sha256": "stale"},  # mismatch -> unconfirmed
    }
    entries = index_entries(mirrors, ledger)
    assert [entry["path"] for entry in entries] == sorted(mirrors)  # deterministic order
    statuses = {entry["path"]: entry["verified_status"] for entry in entries}
    assert statuses["digest/muscle/a.md"] == "confirmed"
    assert statuses["digest/muscle/b.md"] == "unconfirmed"  # hash mismatch renders neutral
    assert statuses["digest/muscle/c.md"] == "unconfirmed"  # never audited renders neutral
    assert entries[0]["scope"] == "general" and entries[1]["scope"] == "device:edge-1"


def test_render_memory_index_lists_paths_status_scope_and_daily():
    entries = [
        {
            "path": "digest/muscle/a.md",
            "description": "Disk cleanup",
            "scope": "general",
            "verified_status": "confirmed",
        },
        {"path": "digest/muscle/b.md", "description": "", "scope": "device:edge-1", "verified_status": "unconfirmed"},
    ]
    text = render_memory_index(entries, ["daily/2026-11-16/incident.md"])
    assert "digest/muscle/a.md" in text and "[confirmed]" in text and "scope=general" in text
    assert "Disk cleanup" in text
    assert "daily/2026-11-16/incident.md" in text
    assert "ops_search" in text  # pointers to where the rest lives


def test_render_memory_index_is_bounded():
    entries = [
        {
            "path": f"digest/muscle/n{i}.md",
            "description": "d" * 80,
            "scope": "general",
            "verified_status": "unconfirmed",
        }
        for i in range(300)
    ]
    text = render_memory_index(entries, [f"daily/2026-01-{i:02d}/n.md" for i in range(1, 11)])
    assert len(text.splitlines()) <= INDEX_MAX_LINES
    assert len(text.encode("utf-8")) <= INDEX_MAX_BYTES
    assert "truncated" in text


def test_render_memory_index_is_deterministic():
    entries = [
        {"path": f"digest/muscle/n{i}.md", "description": f"d{i}", "scope": "general", "verified_status": "confirmed"}
        for i in range(30)
    ]
    first = render_memory_index(entries, ["daily/2026-11-16/a.md"])
    second = render_memory_index(entries, ["daily/2026-11-16/a.md"])
    assert first == second


def test_render_memory_index_empty_mirror_degrades():
    text = render_memory_index([], [])
    assert "no muscle SOPs mirrored yet" in text
    assert "no recent daily notes" in text


def test_recent_daily_pointers_newest_first():
    pointers = recent_daily_pointers(
        ["daily/2026-11-10/a.md", "daily/2026-11-16/b.md", "daily/2026-11-10/a.md"],
        limit=2,
    )
    assert pointers == ["daily/2026-11-16/b.md", "daily/2026-11-10/a.md"]


def _step(tmp_path: Path):
    return OpsReflexIndexStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


@pytest.mark.asyncio
async def test_reflex_index_step_writes_memory_md(tmp_path):
    _write_mirror(tmp_path, "digest/muscle/a.md", "body-a", {"name": "a", "description": "Disk cleanup"})
    daily = tmp_path / "daily" / "2026-11-16"
    daily.mkdir(parents=True)
    (daily / "incident.md").write_text("---\nname: incident\n---\nlog\n", encoding="utf-8")

    response = await _step(tmp_path)(RuntimeContext())
    out = tmp_path / "metadata" / "ops_index" / "MEMORY.md"
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "digest/muscle/a.md" in content and "daily/2026-11-16/incident.md" in content
    meta = response.metadata["reflex_index"]
    assert meta["path"] == "metadata/ops_index/MEMORY.md"
    assert meta["entries"] == 1 and meta["truncated"] is False
    assert response.success is True  # default untouched


@pytest.mark.asyncio
async def test_reflex_index_step_preserves_shared_response(tmp_path):
    """Finding-5 regression: sync failure state must survive the index rebuild."""
    _write_mirror(tmp_path, "digest/muscle/a.md", "body-a", {"name": "a"})
    context = RuntimeContext()
    context.response.success = False
    context.response.answer = "sync failed"

    response = await _step(tmp_path)(context)
    assert response.success is False
    assert response.answer == "sync failed"
    assert response.metadata["reflex_index"]["entries"] == 1
    assert (tmp_path / "metadata" / "ops_index" / "MEMORY.md").is_file()


@pytest.mark.asyncio
async def test_reflex_index_step_empty_mirror_degrades(tmp_path):
    response = await _step(tmp_path)(RuntimeContext())
    out = tmp_path / "metadata" / "ops_index" / "MEMORY.md"
    assert out.is_file()
    assert response.metadata["reflex_index"]["entries"] == 0
    assert response.success is True


@pytest.mark.asyncio
async def test_reflex_index_step_uses_ledger_not_wiped_stamps(tmp_path):
    """Status comes from the hash-gated ledger; a stale mirror stamp is ignored."""
    _write_mirror(
        tmp_path,
        "digest/muscle/a.md",
        "current body",
        {"name": "a", "verified_status": "confirmed"},  # decoration left by an older audit
    )
    ledger_path = audit_status_path(tmp_path, "metadata")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({"digest/muscle/a.md": {"status": "conflict", "body_sha256": body_sha256("current body")}}),
        encoding="utf-8",
    )
    await _step(tmp_path)(RuntimeContext())
    content = (tmp_path / "metadata" / "ops_index" / "MEMORY.md").read_text(encoding="utf-8")
    assert "[conflict]" in content  # ledger wins over the decorative stamp
