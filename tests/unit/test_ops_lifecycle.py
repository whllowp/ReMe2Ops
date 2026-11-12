"""Lifecycle decay: heat formula, cold demotion with tombstones, reversibility."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.lifecycle import (
    OpsDecayScanStep,
    heat_score,
    is_cold,
)


def test_heat_score_recency_decay_and_clamps():
    now = datetime(2026, 11, 13, 12, 0, 0)
    fresh = heat_score(last_used=now.isoformat(), success_rate=1.0, hits=10, now=now)
    assert fresh == pytest.approx(1.0)
    # one half-life on recency only: 0.4*0.5 + 0.3 + 0.3
    half = heat_score(last_used=(now - timedelta(days=14)).isoformat(), success_rate=1.0, hits=10, now=now)
    assert half == pytest.approx(0.4 * 0.5 + 0.6, abs=1e-3)
    stale = heat_score(last_used=(now - timedelta(days=140)).isoformat(), success_rate=0.5, hits=1, now=now)
    assert stale < 0.25
    assert is_cold(stale) and not is_cold(fresh)
    assert heat_score(last_used="garbage", success_rate=0.0, hits=0) == pytest.approx(0.0)
    # access volume without success earns nothing: stale + all-failed stays cold
    degraded = heat_score(last_used=(now - timedelta(days=140)).isoformat(), success_rate=0.0, hits=18, now=now)
    assert degraded == pytest.approx(0.0, abs=1e-3)
    assert is_cold(degraded)


def _store(tmp_path):
    return Mock(spec=BaseFileStore, workspace_path=tmp_path)


def _step(tmp_path):
    step = OpsDecayScanStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), file_store=_store(tmp_path))
    writes = []

    async def fake_run_job(name, **kwargs):
        writes.append((name, kwargs))
        if name == "move":
            src = tmp_path / kwargs["src_path"]
            dst = tmp_path / kwargs["dst_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            src.unlink()
        if name == "frontmatter_update":
            target = tmp_path / kwargs["path"]
            post = frontmatter.loads(target.read_text(encoding="utf-8"))
            post.metadata.update(kwargs["metadata"])
            target.write_text(frontmatter.dumps(post), encoding="utf-8")
        return Mock(success=True, answer="ok")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    broadcasts = []

    async def fake_broadcast(step_, paths, *, reason=""):
        broadcasts.append((paths, reason))

    import reme.steps.ops.lifecycle as lifecycle

    lifecycle_orig = lifecycle.broadcast_muscle_update
    lifecycle.broadcast_muscle_update = fake_broadcast  # type: ignore[assignment]
    return step, writes, broadcasts, lifecycle


def _seed_muscle(tmp_path, records):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    muscle = tmp_path / "digest" / "muscle"
    muscle.mkdir(parents=True, exist_ok=True)
    return muscle


import json


@pytest.mark.asyncio
async def test_decay_scan_demotes_cold_sop_with_tombstone(tmp_path):
    cold_ts = (datetime.now() - timedelta(days=120)).isoformat()
    fresh_ts = datetime.now().isoformat()
    muscle = _seed_muscle(tmp_path, [
        {"ts": cold_ts, "path": "digest/muscle/old.md", "outcome": "solved", "source_device": "d1"},
        {"ts": cold_ts, "path": "digest/muscle/old.md", "outcome": "failed", "source_device": "d1"},
        {"ts": fresh_ts, "path": "digest/muscle/hot.md", "outcome": "solved", "source_device": "d1"},
        {"ts": fresh_ts, "path": "digest/muscle/hot.md", "outcome": "solved", "source_device": "d1"},
    ])
    (muscle / "old.md").write_text("---\nname: old\nscope: general\n---\nbody", encoding="utf-8")
    (muscle / "hot.md").write_text("---\nname: hot\nscope: general\n---\nbody", encoding="utf-8")

    step, writes, broadcasts, lifecycle = _step(tmp_path)
    response = await step(RuntimeContext())

    assert response.success is True
    assert not (muscle / "old.md").exists()  # moved out of the muscle bucket
    demoted = tmp_path / "digest" / "procedure" / "old.md"
    assert demoted.exists()
    post = frontmatter.loads(demoted.read_text(encoding="utf-8"))
    assert post.metadata["lifecycle"] == "demoted"
    assert post.metadata["valid_until"] and post.metadata["demote_reason"]
    assert not (muscle / "hot.md").exists() or (tmp_path / "digest" / "muscle" / "hot.md").exists()
    move_calls = [w for w in writes if w[0] == "move"]
    assert move_calls and move_calls[0][1]["dst_path"] == "digest/procedure/old.md"


@pytest.mark.asyncio
async def test_decay_scan_keeps_hot_sop(tmp_path):
    fresh_ts = datetime.now().isoformat()
    muscle = _seed_muscle(tmp_path, [
        {"ts": fresh_ts, "path": "digest/muscle/hot.md", "outcome": "solved", "source_device": "d1"} for _ in range(5)
    ])
    (muscle / "hot.md").write_text("---\nname: hot\n---\nbody", encoding="utf-8")

    step, writes, _, _ = _step(tmp_path)
    response = await step(RuntimeContext())
    assert response.success is True
    assert (tmp_path / "digest" / "muscle" / "hot.md").exists()
    assert not [w for w in writes if w[0] == "move"]
