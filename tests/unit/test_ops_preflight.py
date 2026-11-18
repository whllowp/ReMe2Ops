"""Retrieval preflight: failure-note warnings on hits, [F] flags in the index.

JARVIS-1 preflight (check past failure causes before executing), closing the
loop with the failure-notes step: failed usage → note on the SOP body → next
retrieval of that SOP must surface the warning. Both surfaces are pure
display: they never touch ranking or weighting (usage boost and the verified
ledger remain the only judges), and an absent failure_count simply disables
the feature.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import FileFrontMatter
from reme.steps.ops.reflex_index import (
    INDEX_MAX_BYTES,
    INDEX_MAX_LINES,
    OpsReflexIndexStep,
    index_entries,
    render_memory_index,
)
from reme.steps.ops.tiered_search import OpsSearchStep, merge_tier_hits

MIRROR = "digest/muscle/disk-cleanup.md"


def _tiers():
    return [
        {"tier": "muscle", "weight": 1.0, "hits": [{"path": MIRROR, "snippet": "runbook"}]},
        {"tier": "daily", "weight": 0.6, "hits": [{"path": "daily/2026-11-18/incident.md", "snippet": "log"}]},
    ]


# ----- merge layer: preflight attaches, ranking untouched ---------------------


def test_preflight_field_attaches_and_never_changes_score():
    plain = merge_tier_hits(_tiers(), limit=5, usage_stats={}, usage_boost=0.05)
    flagged = merge_tier_hits(_tiers(), limit=5, usage_stats={}, usage_boost=0.05, failure_by_path={MIRROR: 2})

    top = flagged[0]
    assert top["preflight"] == {"failure_count": 2}
    assert top["score"] == pytest.approx(plain[0]["score"])  # display only: same score
    assert [item["path"] for item in flagged] == [item["path"] for item in plain]  # ranking identical
    assert "preflight" not in flagged[1]  # hit without failure notes has no field
    assert "preflight" not in plain[0]  # absent map disables the feature entirely


def test_preflight_ignores_zero_and_absent_counts():
    tiers = [{"tier": "muscle", "weight": 1.0, "hits": [{"path": MIRROR}, {"path": "digest/muscle/b.md"}]}]
    merged = merge_tier_hits(tiers, limit=5, usage_stats={}, usage_boost=0.05, failure_by_path={MIRROR: 0})
    assert all("preflight" not in item for item in merged)


# ----- search step: collection, metadata, answer warning line -----------------


def _search_step(tmp_path: Path, store, **kwargs):
    return OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
        cloud_host="127.0.0.1",
        cloud_port=2444,
        **kwargs,
    )


def _stub_cloud(monkeypatch):
    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            return {"success": True, "metadata": {"hits": []}}

    monkeypatch.setattr("reme.steps.ops.tiered_search.CloudClient", StubCloud)


def _search_response(results):
    response = Mock()
    response.success = True
    response.metadata = {"results": results}
    return response


@pytest.mark.asyncio
async def test_search_returns_preflight_for_failing_hit(tmp_path, monkeypatch):
    _stub_cloud(monkeypatch)
    node = Mock(path=MIRROR, front_matter=FileFrontMatter(name="disk-cleanup", failure_count=2))
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    store.get_nodes = AsyncMock(return_value=[node])

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = [{"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}}] if prefix == "digest/muscle" else []
        return _search_response(hits)

    step = _search_step(tmp_path, store, preflight_warnings=True)
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    top = response.metadata["results"][0]
    assert top["preflight"] == {"failure_count": 2}
    assert response.metadata["preflight"] == {MIRROR: 2}
    assert "[preflight] 2 failure note(s)" in response.answer  # warning line in the answer
    assert "## Failure Notes" in response.answer
    assert top["score"] == pytest.approx(0.5)  # muscle rank-1 weight only: not weighted


@pytest.mark.asyncio
async def test_search_without_failure_frontmatter_has_no_preflight(tmp_path, monkeypatch):
    _stub_cloud(monkeypatch)
    node = Mock(path=MIRROR, front_matter=FileFrontMatter(name="disk-cleanup"))
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    store.get_nodes = AsyncMock(return_value=[node])

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = [{"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}}] if prefix == "digest/muscle" else []
        return _search_response(hits)

    step = _search_step(tmp_path, store, preflight_warnings=True)
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    assert "preflight" not in response.metadata["results"][0]
    assert "preflight" not in response.metadata
    assert "[preflight]" not in response.answer


@pytest.mark.asyncio
async def test_search_preflight_switch_off_is_fully_inert(tmp_path, monkeypatch):
    """Default off: failure frontmatter present but no preflight anywhere."""
    _stub_cloud(monkeypatch)
    node = Mock(path=MIRROR, front_matter=FileFrontMatter(name="disk-cleanup", failure_count=3))
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    store.get_nodes = AsyncMock(return_value=[node])

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = [{"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}}] if prefix == "digest/muscle" else []
        return _search_response(hits)

    step = _search_step(tmp_path, store)  # no preflight_warnings kwarg
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    assert "preflight" not in response.metadata["results"][0]
    assert "preflight" not in response.metadata
    assert "[preflight]" not in response.answer


# ----- reflex index: [F] flag within the same budget --------------------------


def test_index_entries_marks_has_failures():
    mirrors = {
        MIRROR: {"description": "Disk cleanup", "scope": "general", "body": "body-a", "failure_count": 2},
        "digest/muscle/b.md": {"description": "Log rotate", "scope": "general", "body": "body-b"},
        "digest/muscle/c.md": {"description": "", "scope": "general", "body": "body-c", "failure_count": 0},
    }
    entries = index_entries(mirrors, {})
    marks = {entry["path"]: entry["has_failures"] for entry in entries}
    assert marks[MIRROR] is True
    assert marks["digest/muscle/b.md"] is False  # absent field -> no failures
    assert marks["digest/muscle/c.md"] is False  # zero count -> no failures


def test_render_memory_index_flags_failures_with_line_budget_unchanged():
    flagged = [
        {"path": MIRROR, "description": "Disk cleanup", "scope": "general", "verified_status": "unconfirmed", "has_failures": True},
        {"path": "digest/muscle/b.md", "description": "Log rotate", "scope": "general", "verified_status": "confirmed", "has_failures": False},
    ]
    plain = [
        {k: v for k, v in entry.items() if k != "has_failures"} for entry in flagged
    ]
    text_flagged = render_memory_index(flagged, [])
    text_plain = render_memory_index(plain, [])
    assert "[F]" in text_flagged and "digest/muscle/disk-cleanup.md]] [unconfirmed]" in text_flagged
    assert "[F]" not in text_plain
    assert len(text_flagged.splitlines()) == len(text_plain.splitlines())  # same line count
    assert len(text_flagged.splitlines()) <= INDEX_MAX_LINES
    assert len(text_flagged.encode("utf-8")) <= INDEX_MAX_BYTES


def test_render_memory_index_flagged_entries_stay_within_budget():
    entries = [
        {
            "path": f"digest/muscle/n{i}.md",
            "description": "d" * 80,
            "scope": "general",
            "verified_status": "unconfirmed",
            "has_failures": True,  # every entry flagged: worst case for the budget
        }
        for i in range(300)
    ]
    text = render_memory_index(entries, [f"daily/2026-01-{i:02d}/n.md" for i in range(1, 11)])
    assert len(text.splitlines()) <= INDEX_MAX_LINES
    assert len(text.encode("utf-8")) <= INDEX_MAX_BYTES
    assert text.count("[F]") >= 1


@pytest.mark.asyncio
async def test_reflex_index_step_renders_f_flag_from_frontmatter(tmp_path):
    mirror = tmp_path / MIRROR
    mirror.parent.mkdir(parents=True)
    mirror.write_text(
        frontmatter.dumps(frontmatter.Post("body with notes", name="disk-cleanup", description="Disk cleanup", failure_count=2)),
        encoding="utf-8",
    )
    step = OpsReflexIndexStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )
    response = await step(RuntimeContext())

    content = (tmp_path / "metadata" / "ops_index" / "MEMORY.md").read_text(encoding="utf-8")
    assert "[F]" in content and "[unconfirmed]" in content
    assert response.metadata["reflex_index"]["lines"] <= INDEX_MAX_LINES
    assert response.metadata["reflex_index"]["bytes"] <= INDEX_MAX_BYTES


# ----- config wiring ----------------------------------------------------------


def test_edge_search_carries_preflight_switch():
    edge = _load_config("ops_edge.yaml")
    search = edge["jobs"]["ops_search"]["steps"][0]
    assert search["backend"] == "ops_search_step"
    assert search["preflight_warnings"] is True
