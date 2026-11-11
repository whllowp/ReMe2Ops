"""Dream-RSI replay simulator: session worlds, policy scoring, deployment loop."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.evolution import OpsPolicyEvolveStep, OpsReplayStep, default_candidate_grid
from reme.steps.ops.replay import (
    build_session_trees,
    load_deployed_policy_file,
    policy_params,
    replay_once,
    replay_score,
    save_deployed_policy,
)


def _record(ts, path, outcome, session, device="d1", query="disk full"):
    return {"ts": ts, "path": path, "outcome": outcome, "session_id": session,
            "query": query, "source_device": device}


def test_build_session_trees_groups_and_sorts():
    records = [
        _record("t3", "b.md", "failed", "s1"),
        _record("t1", "a.md", "solved", "s1"),
        _record("t2", "a.md", "solved", "s1"),
        _record("t1", "c.md", "solved", "s2", query="other"),
        {"ts": "t9", "path": "orphan.md", "outcome": "solved"},  # no session/query -> dropped
    ]
    trees = build_session_trees(records)
    assert len(trees) == 2
    assert [r["ts"] for r in trees[0]] == ["t1", "t2", "t3"]
    assert trees[1][0]["path"] == "c.md"


def test_replay_once_promotes_and_counts_reflex_hits():
    # Session 1: two solves promote the SOP; session 2: the reflex serves a solve.
    trees = build_session_trees([
        _record("t1", "a.md", "solved", "s1"),
        _record("t2", "a.md", "solved", "s1"),
        _record("t3", "a.md", "solved", "s2"),
        _record("t4", "a.md", "failed", "s3"),
    ])
    params = policy_params({"min_solved": 2, "min_success_rate": 0.5, "score_bar": 0.5})
    counts = replay_once(trees, params)
    assert counts["promotions"] == 1
    assert counts["reflex_hits"] == 1  # t3 served from would-be muscle
    assert counts["false_reflex"] == 1  # t4 failed on the muscle path
    assert counts["muscle_size"] == 1


def test_replay_once_strict_policy_never_promotes():
    trees = build_session_trees([_record("t1", "a.md", "solved", "s1")])
    params = policy_params({"min_solved": 2})
    counts = replay_once(trees, params)
    assert counts["promotions"] == 0 and counts["reflex_hits"] == 0


def test_replay_score_sweeps_bar_and_penalizes_false_reflex():
    trees = build_session_trees([
        _record("t1", "a.md", "solved", "s1"),
        _record("t2", "a.md", "solved", "s1"),
        _record("t3", "a.md", "solved", "s2"),
    ])
    result = replay_score(trees, policy_params({"min_solved": 1, "score_bar": 0.4}))
    assert result["frontier"], "lenient sweep must expose a frontier point"
    assert 0.0 <= result["hypervolume"] <= 1.0
    assert result["false_rate"] == 0.0  # no muscle-path failures in this world


def test_candidate_grid_dedupes_and_includes_deployed():
    grid = default_candidate_grid({"min_solved": 2, "name": "deployed"})
    assert grid[0]["name"] == "deployed"
    markers = {json.dumps(policy_params(c), sort_keys=True) for c in grid}
    assert len(markers) == len(grid)  # no duplicate param sets
    assert len(grid) <= 12


def _store(tmp_path):
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    return store


def _seed_worlds(tmp_path, sessions=6):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for s in range(sessions):
        records.append(_record(f"t{s}1", "digest/procedure/star.md", "solved", f"s{s}"))
        records.append(_record(f"t{s}2", "digest/procedure/star.md", "solved", f"s{s}"))
        records.append(_record(f"t{s}3", "digest/procedure/star.md", "solved", f"s{s+100}"))
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


@pytest.mark.asyncio
async def test_replay_then_evolve_deploys_best_policy(tmp_path):
    _seed_worlds(tmp_path)
    ctx = ApplicationContext(workspace_dir=str(tmp_path))
    replay = OpsReplayStep(app_context=ctx, file_store=_store(tmp_path))
    await replay(RuntimeContext())
    assert replay.context["replay_redeploy"] is True  # no deployed policy yet

    evolve = OpsPolicyEvolveStep(app_context=ctx, file_store=_store(tmp_path))
    response = await evolve(RuntimeContext(
        replay_winner=replay.context["replay_winner"],
        replay_redeploy=replay.context["replay_redeploy"],
    ))
    assert response.success is True
    record = load_deployed_policy_file(tmp_path, "metadata")
    assert record["policy_version"] == 1
    assert record["params"]["name"] == replay.context["replay_winner"]["name"]


@pytest.mark.asyncio
async def test_replay_keeps_better_deployed_policy(tmp_path):
    save_deployed_policy(tmp_path, "metadata", {
        "params": policy_params({"min_solved": 1, "name": "champion"}),
        "replay_score": 1.0,
        "policy_version": 3,
    })
    _seed_worlds(tmp_path, sessions=2)
    ctx = ApplicationContext(workspace_dir=str(tmp_path))
    replay = OpsReplayStep(app_context=ctx, file_store=_store(tmp_path))
    await replay(RuntimeContext())

    assert replay.context["replay_redeploy"] is False  # champion score 1.0 unbeatable
    evolve = OpsPolicyEvolveStep(app_context=ctx, file_store=_store(tmp_path))
    response = await evolve(RuntimeContext(
        replay_winner=replay.context["replay_winner"],
        replay_redeploy=replay.context["replay_redeploy"],
    ))
    record = load_deployed_policy_file(tmp_path, "metadata")
    assert record["policy_version"] == 3  # untouched
    assert "Kept deployed policy" in response.answer


@pytest.mark.asyncio
async def test_replay_without_worlds_fails_cleanly(tmp_path):
    ctx = ApplicationContext(workspace_dir=str(tmp_path))
    replay = OpsReplayStep(app_context=ctx, file_store=_store(tmp_path))
    response = await replay(RuntimeContext())
    assert response.success is False
    assert "No replay worlds" in response.answer
