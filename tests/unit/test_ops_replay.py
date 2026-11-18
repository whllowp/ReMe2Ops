"""Dream-RSI replay simulator: session worlds, policy scoring, deployment loop.

Round 9 extends the genome: with ``mutate_weights`` on, the λ mix and
``solved_cap`` join the dreaming mutation space (Sage-Taiyi cognitive
repertoire — the reusable unit is the method, so the method space should be
at least as wide as the threshold space). All mutation stays deterministic
(round-seeded RNG, fixed draw order), the λ sum stays exactly 1 (Tchebycheff
decomposition semantics), and the deployed policy never regresses.
"""

import json
import random
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.evolution import (
    OpsPolicyEvolveStep,
    OpsReplayStep,
    default_candidate_grid,
    mutate_params,
    normalize_weights,
)
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


def test_replay_score_threads_independent_only_to_every_bar():
    """The deployed-policy evaluation runs on the same counting basis as the
    live scan: per-bar counts must equal an explicit replay_once with the same
    flag, and the deduped view must be present under the flag only."""
    trees = build_session_trees([
        _record("t1", "a.md", "solved", "s1"),
        _record("t2", "a.md", "solved", "s1"),
        _record("t3", "a.md", "solved", "s1"),
        _record("t4", "a.md", "failed", "s2"),
    ])
    params = policy_params({"min_solved": 1, "min_success_rate": 0.5, "score_bar": 0.4})
    result = replay_score(trees, params, independent_only=True)
    for bar, counts in result["counts"].items():
        swept = dict(params, score_bar=float(bar))
        expected = replay_once(trees, swept, independent_only=True)
        assert counts == expected
        assert "distinct_solved_by_path" in counts  # deduped view present
    assert "distinct_solved_by_path" not in replay_score(trees, params)["counts"]["0.4"]  # flag off: raw view


def test_candidate_grid_dedupes_and_includes_deployed():
    grid = default_candidate_grid({"min_solved": 2, "name": "deployed"})
    assert grid[0]["name"] == "deployed"
    markers = {json.dumps(policy_params(c), sort_keys=True) for c in grid}
    assert len(markers) == len(grid)  # no duplicate param sets
    assert len(grid) <= 12


# ----- round 9: policy genome (λ weights + solved_cap) -------------------------


def test_mutate_params_weights_sum_to_one_and_stay_positive():
    base = policy_params({"min_solved": 2})
    for seed in range(24):
        candidate = mutate_params(base, random.Random(seed), mutate_weights=True)
        weights = candidate["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)  # Tchebycheff semantics intact
        assert all(value > 0 for value in weights.values())


def test_normalize_weights_alone_is_normalized():
    mix = normalize_weights({"rate": 0.4 * 1.17, "solved": 0.3 * 0.86, "breadth": 0.3 * 1.05})
    assert sum(mix.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value > 0 for value in mix.values())


def test_mutate_params_reproducible_per_seed_and_cap_bounded():
    base = policy_params({"min_solved": 2})
    first = mutate_params(base, random.Random(11), mutate_weights=True)
    second = mutate_params(base, random.Random(11), mutate_weights=True)
    assert first == second  # same round seed -> same candidate set
    different = mutate_params(base, random.Random(12), mutate_weights=True)
    assert different != first
    assert 1.0 <= first["solved_cap"] <= 50.0


def test_mutate_params_without_flag_keeps_weights_and_cap():
    base = policy_params({"min_solved": 2})
    candidate = mutate_params(base, random.Random(7))  # default: genome unchanged
    assert candidate["weights"] == base["weights"]
    assert candidate["solved_cap"] == base["solved_cap"]


def test_candidate_grid_with_weights_leads_the_pool():
    grid = default_candidate_grid({"min_solved": 2, "name": "deployed"}, mutate_weights=True)
    assert grid[0]["name"] == "deployed"  # deployed always first / never regress constraint
    assert any(candidate["weights"] != grid[0]["weights"] for candidate in grid)  # weight mixes present
    assert any(candidate["solved_cap"] != grid[0]["solved_cap"] for candidate in grid)  # cap candidates present
    markers = {json.dumps(policy_params(c), sort_keys=True) for c in grid}
    assert len(markers) == len(grid)
    assert len(grid) <= 12
    # flag off keeps the legacy grid: no weight or cap variants
    legacy = default_candidate_grid({"min_solved": 2, "name": "deployed"})
    assert all(candidate["weights"] == legacy[0]["weights"] for candidate in legacy)
    assert all(candidate["solved_cap"] == legacy[0]["solved_cap"] for candidate in legacy)


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


@pytest.mark.asyncio
async def test_round_seed_reproducible_with_weights(tmp_path):
    def without_ts(curve):
        return [{key: value for key, value in entry.items() if key != "ts"} for entry in curve]

    _seed_worlds(tmp_path)
    curves = []
    for _ in range(2):
        replay = OpsReplayStep(
            app_context=ApplicationContext(workspace_dir=str(tmp_path)),
            file_store=_store(tmp_path),
            rounds=3,
            mutate_weights=True,
        )
        response = await replay(RuntimeContext())
        curves.append(response.metadata["replay"]["rounds"])

    # same round seed -> identical candidate sets and scores (only the wall
    # clock `ts` differs between the two runs)
    assert without_ts(curves[0]) == without_ts(curves[1])
    assert all(entry["candidates"] == 4 for entry in curves[0])  # incumbent + 3 mutants
    # winners come from the pool naming scheme (incumbent or a mutant)
    assert all(
        entry["best_candidate"] == "incumbent" or entry["best_candidate"].startswith("mut-")
        for entry in curves[0]
    )


@pytest.mark.asyncio
async def test_rounds1_with_weights_stays_compatible(tmp_path):
    _seed_worlds(tmp_path, sessions=2)
    replay = OpsReplayStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        rounds=1,
        mutate_weights=True,
    )
    response = await replay(RuntimeContext())

    assert response.success is True
    rounds = response.metadata["replay"]["rounds"]
    assert len(rounds) == 1
    winner_weights = response.metadata["replay"]["winner"]["params"]["weights"]
    assert sum(winner_weights.values()) == pytest.approx(1.0, abs=1e-6)  # decomposition semantics intact


@pytest.mark.asyncio
async def test_deployed_policy_never_regresses_with_weights(tmp_path):
    """The genome is wider, but the never-regress constraint still holds."""
    save_deployed_policy(tmp_path, "metadata", {
        "params": policy_params({"min_solved": 1, "name": "champion"}),
        "replay_score": 1.0,
        "policy_version": 3,
    })
    _seed_worlds(tmp_path, sessions=2)
    replay = OpsReplayStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        rounds=2,
        mutate_weights=True,
    )
    await replay(RuntimeContext())
    assert replay.context["replay_redeploy"] is False  # champion score 1.0 unbeatable

    evolve = OpsPolicyEvolveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), file_store=_store(tmp_path))
    response = await evolve(RuntimeContext(
        replay_winner=replay.context["replay_winner"],
        replay_redeploy=replay.context["replay_redeploy"],
    ))
    record = load_deployed_policy_file(tmp_path, "metadata")
    assert record["policy_version"] == 3  # untouched: pi never regresses
    assert "Kept deployed policy" in response.answer


@pytest.mark.asyncio
async def test_replay_step_threads_independent_only(tmp_path, monkeypatch):
    """Round-10 review fix: the dreaming evaluation must run on the same
    session-deduped evidence basis the live promotion scan uses."""
    import reme.steps.ops.evolution as evolution_module

    real_replay_score = evolution_module.replay_score
    seen_flags: list[bool] = []

    def spy_replay_score(trees, params, *, independent_only=False, parallelism_penalty=0.1):
        seen_flags.append(independent_only)
        return real_replay_score(
            trees,
            params,
            independent_only=independent_only,
            parallelism_penalty=parallelism_penalty,
        )

    monkeypatch.setattr(evolution_module, "replay_score", spy_replay_score)
    _seed_worlds(tmp_path, sessions=2)
    replay = OpsReplayStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        rounds=1,
        independent_only=True,
    )
    response = await replay(RuntimeContext(policies=[{"min_solved": 1, "name": "candidate"}]))

    assert response.success is True
    assert seen_flags == [True]  # explicit pool: exactly one replay_score call, flag threaded
    winner = response.metadata["replay"]["winner"]["params"]["weights"]
    assert sum(winner.values()) == pytest.approx(1.0, abs=1e-6)

    # default: flag off (raw counting, the pre-round-3 behaviour)
    _seed_worlds(tmp_path, sessions=1)
    replay2 = OpsReplayStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        rounds=1,
    )
    seen_flags.clear()

    def recorder(trees, params, **kwargs):
        seen_flags.append(bool(kwargs.get("independent_only", False)))
        return real_replay_score(trees, params, independent_only=kwargs.get("independent_only", False))

    monkeypatch.setattr(evolution_module, "replay_score", recorder)
    await replay2(RuntimeContext(policies=[{"min_solved": 1, "name": "candidate"}]))
    assert seen_flags == [False]
