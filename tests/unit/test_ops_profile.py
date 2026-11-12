"""SHARP device profile: statistics, bounded iterative updates, provenance."""

import json
from unittest.mock import Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.profile import OpsProfileStep, build_statistics, parse_ts


def test_build_statistics_counts_and_histogram():
    stats = build_statistics([
        {"ts": "2026-11-12T02:30:00", "path": "a.md", "outcome": "solved", "source_device": "d1"},
        {"ts": "2026-11-12T03:10:00", "path": "a.md", "outcome": "solved", "source_device": "d1"},
        {"ts": "2026-11-12T04:00:00", "path": "b.md", "outcome": "failed", "source_device": "d1"},
        {"ts": "nope", "path": "b.md", "outcome": "solved", "source_device": "d1"},
    ])
    assert stats["solved"] == 3 and stats["failed"] == 1
    assert stats["top_paths"][0] == {"path": "a.md", "hits": 2}
    assert stats["hour_histogram"] == {"02": 1, "03": 1, "04": 1}


def test_parse_ts_handles_garbage():
    assert parse_ts("2026-11-12T02:30:00") is not None
    assert parse_ts("garbage") is None


class FakeWrapper(BaseAgentWrapper):
    """Shrinking profile: proves the rewrite honors the compression contract."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def reply(self, inputs, **kwargs):
        self.calls += 1
        return {"status": "completed", "error": None,
                "result": json.dumps({"profile": f"## Device profile v{self.calls}\nTop alert: disk.\n"})}


def _step(tmp_path, wrapper=None):
    kwargs = {"agent_wrapper": wrapper} if wrapper is not None else {}
    return OpsProfileStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        device_id="edge-a",
        **kwargs,
    )


def _seed_usage(tmp_path, rounds=5):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for r in range(rounds * 3):
            fh.write(json.dumps({
                "ts": f"2026-11-12T0{r % 6}:00:00",
                "path": f"digest/muscle/runbook-{r % 3}.md",
                "outcome": "solved" if r % 4 else "failed",
                "source_device": "edge-a",
            }) + "\n")


@pytest.mark.asyncio
async def test_profile_fallback_is_bounded_and_iterates(tmp_path):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    step = _step(tmp_path)  # no LLM components: deterministic S-only stage
    sizes = []
    for round_index in range(10):
        # Each round delivers fresh device events (real iteration semantics).
        with usage.open("a", encoding="utf-8") as fh:
            for r in range(3):
                fh.write(json.dumps({
                    "ts": f"2026-11-12T{round_index:02d}:{r:02d}:00",
                    "path": f"digest/muscle/runbook-{r % 3}.md",
                    "outcome": "solved" if r % 2 else "failed",
                    "source_device": "edge-a",
                }) + "\n")
        response = await step(RuntimeContext())
        assert response.success is True
        assert response.metadata["profile"]["stage"] == "statistics"
        assert response.metadata["profile"]["version"] == round_index + 1
        profile_path = tmp_path / "metadata" / "ops_profile" / "edge-a.md"
        post = frontmatter.loads(profile_path.read_text(encoding="utf-8"))
        sizes.append(len(post.content))
        expected_from = "seed" if round_index == 0 else f"metadata/ops_profile/edge-a.md#v{round_index}"
        assert post.metadata["derived_from"] == expected_from
        assert post.metadata["model_version"] == "ops-profile-v1"
        assert len(post.content) <= 4000
    assert len(sizes) == 10
    assert len({s for s in sizes}) <= 3  # bounded: does not grow linearly with rounds


@pytest.mark.asyncio
async def test_profile_agent_stage_uses_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.profile.llm_available", lambda step: True)
    _seed_usage(tmp_path)
    wrapper = FakeWrapper()
    step = _step(tmp_path, wrapper)
    response = await step(RuntimeContext())
    assert response.metadata["profile"]["stage"] == "agent"
    assert wrapper.calls == 1
    post = frontmatter.loads((tmp_path / "metadata" / "ops_profile" / "edge-a.md").read_text(encoding="utf-8"))
    assert post.content.startswith("## Device profile")


@pytest.mark.asyncio
async def test_profile_without_new_events_is_noop(tmp_path):
    _seed_usage(tmp_path)
    step = _step(tmp_path)
    await step(RuntimeContext())
    response = await step(RuntimeContext())  # cursor advanced: nothing new
    assert response.metadata["profile"]["updated"] is False
