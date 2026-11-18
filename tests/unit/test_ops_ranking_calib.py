"""Ranking-weight dreaming calibration and the cloud->edge deployment channel.

MATM §3.2 LTRT ranks retrievals by marginal utility (solved with vs without),
and the labels come free from the incremental build process — here the offline
replay records which SOP actually solved each event, so the boost-vector grid
is scored by the deterministic top-1 proxy objective. The winner deploys to
metadata/ops_policy/ranking.json on the cloud, travels to edges piggybacked on
the existing muscle_list pull, lands as the edge snapshot via muscle_sync, and
the search step reads it with the promote.py precedence contract: explicit
kwargs > deployed snapshot > built-in defaults.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.muscle_list import OpsMuscleListStep, load_ranking_policy
from reme.steps.ops.muscle_sync import OpsMuscleSyncStep
from reme.steps.ops.ranking_calib import (
    CALIBRATION_GRID,
    OpsRankingCalibrateStep,
    boost_l1,
    calibrate_ranking_params,
    pareto_front,
    ranking_policy_path,
    replay_ranking_objective,
)
from reme.steps.ops.replay import build_session_trees
from reme.steps.ops.tiered_search import OpsSearchStep, load_ranking_boosts

HERO = "digest/muscle/z-hero.md"  # sorts last: ties would not favour it
SIDE = "digest/muscle/a-side.md"  # sorts first: wins name-tie breaks
MIRROR = "digest/muscle/disk-cleanup.md"  # search-step hit path

GRID = [
    {"usage_boost": 0.0, "confirmed_boost": 0.0, "conflict_penalty": 0.0},
    {"usage_boost": 0.1, "confirmed_boost": 0.05, "conflict_penalty": 0.2},
]


def _record(ts, path, outcome, session="s1"):
    return {"ts": ts, "path": path, "outcome": outcome, "session_id": session}


def _step(tmp_path: Path):
    return OpsRankingCalibrateStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


def _write_snapshot(tmp_path: Path, payload: dict) -> None:
    path = ranking_policy_path(tmp_path, "metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# ----- pure calibration --------------------------------------------------------


def _contested_world() -> list[list[dict]]:
    """a-side solves first, z-hero catches up, a-side then accumulates failures.

    With conflict_penalty 0 the name tie-break keeps a-side top at the tied
    t07 event; with penalty 0.2 its failures sink it and z-hero takes top-1 —
    so the objective separates the grid vectors (the proxy floors are looser
    than live promotion on purpose: failing paths must stay rankable).
    """
    return build_session_trees(
        [
            _record("t01", SIDE, "solved"),
            _record("t02", SIDE, "solved"),
            _record("t03", SIDE, "solved"),
            _record("t04", HERO, "solved"),
            _record("t05", HERO, "solved"),
            _record("t06", SIDE, "failed"),
            _record("t07", HERO, "solved"),
            _record("t08", SIDE, "failed"),
            _record("t09", HERO, "solved"),
            _record("t10", HERO, "solved"),
            _record("t11", SIDE, "solved"),
        ],
    )


def test_replay_ranking_objective_depends_on_the_boost_vector():
    world = _contested_world()
    zero = replay_ranking_objective(world, {"usage_boost": 0.0, "confirmed_boost": 0.0, "conflict_penalty": 0.0})
    harsh = replay_ranking_objective(world, {"usage_boost": 0.1, "confirmed_boost": 0.0, "conflict_penalty": 0.2})
    assert zero["solved_events"] == harsh["solved_events"] > 0
    assert harsh["objective"] > zero["objective"]  # failures sink a-side under the harsh vector


def test_calibration_monotone_non_worse_for_high_solved_first_worlds():
    """A world where one SOP keeps solving: higher usage_boost never hurts."""
    world = build_session_trees(
        [
            _record("t01", HERO, "solved"),
            _record("t02", HERO, "solved"),
            _record("t03", HERO, "solved"),
            _record("t04", HERO, "solved"),
            _record("t05", HERO, "solved"),
            _record("t06", SIDE, "solved"),
            _record("t07", SIDE, "solved"),
        ],
    )
    objectives = [
        replay_ranking_objective(world, {"usage_boost": boost, "confirmed_boost": 0.05, "conflict_penalty": 0.15})["objective"]
        for boost in (0.0, 0.05, 0.1, 0.2)
    ]
    assert objectives == sorted(objectives, reverse=False)  # monotone non-decreasing
    assert objectives[-1] >= objectives[0]


def test_calibrate_ranking_params_pareto_front_and_winner():
    result = calibrate_ranking_params(_contested_world(), GRID)
    assert result["evaluated"] == len(GRID)
    assert result["frontier"], "a non-dominated front must exist"
    assert result["winner"] == result["frontier"][0]
    for entry in result["frontier"]:
        assert entry["boost_l1"] == pytest.approx(boost_l1(entry["boosts"]))
        assert set(entry["boosts"]) == {"usage_boost", "confirmed_boost", "conflict_penalty"}
    # the zero vector is dominated by any strictly better objective
    assert all(entry["objective"] >= 0 for entry in result["frontier"])
    assert result["solved_events"] > 0


def test_pareto_front_prefers_same_utility_less_intervention():
    candidates = [
        {"boosts": {"usage_boost": 0.0}, "objective": 0.5, "boost_l1": 0.0},
        {"boosts": {"usage_boost": 0.2}, "objective": 0.5, "boost_l1": 0.2},  # dominated: same utility
        {"boosts": {"usage_boost": 0.2}, "objective": 0.8, "boost_l1": 0.2},  # better utility
    ]
    front = pareto_front(candidates)
    assert {entry["objective"] for entry in front} == {0.5, 0.8}
    assert front[0]["objective"] == 0.8  # head: best utility, then smallest norm


# ----- calibrate step ----------------------------------------------------------


def _seed_cloud_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _seed_edge_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"  # the search step reads edge records
    usage.parent.mkdir(parents=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.mark.asyncio
async def test_calibrate_step_writes_ranking_policy_file(tmp_path):
    _seed_cloud_usage(tmp_path, [
        _record("2026-11-18T01:00:00", HERO, "solved"),
        _record("2026-11-18T02:00:00", HERO, "solved"),
        _record("2026-11-18T03:00:00", SIDE, "solved"),
        _record("2026-11-18T04:00:00", SIDE, "solved"),
        _record("2026-11-18T05:00:00", SIDE, "solved"),
        _record("2026-11-18T06:00:00", SIDE, "failed"),
        _record("2026-11-18T07:00:00", HERO, "solved"),
    ])
    response = await _step(tmp_path)(RuntimeContext())

    meta = response.metadata["ranking_calib"]
    assert meta["skipped"] == ""
    assert meta["path"] == "metadata/ops_policy/ranking.json"
    snapshot = load_ranking_policy(tmp_path, "metadata")
    assert set(snapshot["deployed"]) == {"usage_boost", "confirmed_boost", "conflict_penalty"}
    assert snapshot["worlds"] >= 1 and snapshot["calibrated_at"]
    assert meta["deployed"] == snapshot["deployed"]
    assert meta["evaluated"] == len(CALIBRATION_GRID)


@pytest.mark.asyncio
async def test_calibrate_step_without_usage_writes_nothing(tmp_path):
    (tmp_path / "metadata").mkdir()
    response = await _step(tmp_path)(RuntimeContext())

    assert not ranking_policy_path(tmp_path, "metadata").exists()
    assert "no replay worlds" in response.metadata["ranking_calib"]["skipped"]
    assert response.success is True  # shared-response contract untouched


@pytest.mark.asyncio
async def test_calibrate_step_without_muscle_solved_events_skips(tmp_path):
    _seed_cloud_usage(tmp_path, [_record("2026-11-18T01:00:00", "digest/procedure/one-off.md", "solved")])
    response = await _step(tmp_path)(RuntimeContext())

    assert not ranking_policy_path(tmp_path, "metadata").exists()
    assert "no would-be-muscle solved events" in response.metadata["ranking_calib"]["skipped"]


# ----- muscle_list deployment channel ------------------------------------------


def _muscle_list_step(tmp_path: Path):
    return OpsMuscleListStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


@pytest.mark.asyncio
async def test_muscle_list_carries_policy_block(tmp_path):
    mirror = tmp_path / "digest" / "muscle" / "a.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(frontmatter.dumps(frontmatter.Post("runbook", name="a", description="d")), encoding="utf-8")
    payload = {"deployed": {"usage_boost": 0.1, "confirmed_boost": 0.05, "conflict_penalty": 0.15}, "worlds": 7}
    _write_snapshot(tmp_path, payload)

    response = await _muscle_list_step(tmp_path)(RuntimeContext())

    assert response.metadata["policy"]["ranking"] == payload  # verbatim piggyback
    assert [f["path"] for f in response.metadata["files"]] == ["digest/muscle/a.md"]  # list untouched
    assert load_ranking_policy(tmp_path, "metadata") == payload


@pytest.mark.asyncio
async def test_muscle_list_without_policy_file_has_no_policy_block(tmp_path):
    mirror = tmp_path / "digest" / "muscle" / "a.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(frontmatter.dumps(frontmatter.Post("runbook", name="a")), encoding="utf-8")

    response = await _muscle_list_step(tmp_path)(RuntimeContext())
    assert "policy" not in response.metadata
    assert len(response.metadata["files"]) == 1

    # a corrupted policy file degrades to no block, never an error
    _write_snapshot(tmp_path, {})
    ranking_policy_path(tmp_path, "metadata").write_text("{broken", encoding="utf-8")
    response2 = await _muscle_list_step(tmp_path)(RuntimeContext())
    assert "policy" not in response2.metadata


# ----- muscle_sync snapshot landing --------------------------------------------


def _sync_step(tmp_path: Path):
    return OpsMuscleSyncStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


@pytest.mark.asyncio
async def test_sync_lands_edge_snapshot_from_policy_block(tmp_path, monkeypatch):
    payload = {"deployed": {"usage_boost": 0.1, "confirmed_boost": 0.05, "conflict_penalty": 0.15}, "worlds": 3}

    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload_):
            return {"success": True, "metadata": {"files": [], "policy": {"ranking": payload}}}

    monkeypatch.setattr("reme.steps.ops.muscle_sync.CloudClient", StubCloud)
    response = await _sync_step(tmp_path)(RuntimeContext())

    snapshot = load_ranking_policy(tmp_path, "metadata")
    assert snapshot == payload  # verbatim edge snapshot
    assert response.metadata["policy_synced"] is True
    assert response.success is True


@pytest.mark.asyncio
async def test_sync_keeps_existing_snapshot_when_cloud_absent(tmp_path, monkeypatch):
    existing = {"deployed": {"usage_boost": 0.2, "confirmed_boost": 0.05, "conflict_penalty": 0.15}}
    _write_snapshot(tmp_path, existing)

    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload_):
            # no policy block at all (cloud has not calibrated yet)
            return {"success": True, "metadata": {"files": []}}

    monkeypatch.setattr("reme.steps.ops.muscle_sync.CloudClient", StubCloud)
    response = await _sync_step(tmp_path)(RuntimeContext())

    assert load_ranking_policy(tmp_path, "metadata") == existing  # untouched
    assert response.metadata["policy_synced"] is False

    # a malformed ranking payload also leaves the snapshot untouched
    class MalformedCloud(StubCloud):
        async def post(self, action, payload_):
            return {"success": True, "metadata": {"files": [], "policy": {"ranking": "not-a-dict"}}}

    monkeypatch.setattr("reme.steps.ops.muscle_sync.CloudClient", MalformedCloud)
    response2 = await _sync_step(tmp_path)(RuntimeContext())
    assert load_ranking_policy(tmp_path, "metadata") == existing
    assert response2.metadata["policy_synced"] is False


# ----- search boost precedence -------------------------------------------------


def _search_step(tmp_path: Path, **kwargs):
    return OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        cloud_host="127.0.0.1",
        cloud_port=2444,
        **kwargs,
    )


def _wire_search(monkeypatch, tmp_path: Path, step: OpsSearchStep):
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

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = [{"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}}] if prefix == "digest/muscle" else []
        response = Mock()
        response.success = True
        response.metadata = {"results": hits}
        return response

    step.run_job = AsyncMock(side_effect=fake_run_job)
    return step


@pytest.mark.asyncio
async def test_search_uses_builtin_defaults_when_no_snapshot(tmp_path, monkeypatch):
    _wire_search(monkeypatch, tmp_path, _search_step(tmp_path))
    response = await _search_step(tmp_path)(RuntimeContext(query="disk full"))

    # a solved history would add usage_boost per solve: the built-in default
    # must reproduce the shipped yaml bootstrap value of 0.05 exactly
    boosts = response.metadata["ranking_boosts"]
    assert boosts["usage_boost"] == 0.05 and boosts["confirmed_boost"] == 0.05
    assert boosts["conflict_penalty"] == 0.15
    assert set(boosts["sources"].values()) == {"builtin"}


@pytest.mark.asyncio
async def test_search_prefers_deployed_snapshot_over_builtin(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, {"deployed": {"usage_boost": 0.2, "confirmed_boost": 0.02, "conflict_penalty": 0.3}})
    _seed_edge_usage(tmp_path, [
        {"ts": "2026-11-18T01:00:00", "path": MIRROR, "outcome": "solved", "session_id": "s1"},
    ])
    step = _wire_search(monkeypatch, tmp_path, _search_step(tmp_path))
    response = await step(RuntimeContext(query="disk full"))

    top = response.metadata["results"][0]
    assert top["usage"]["solved"] == 1
    assert top["score"] == pytest.approx(0.5 + 0.2)  # snapshot usage_boost applied
    boosts = response.metadata["ranking_boosts"]
    assert boosts["usage_boost"] == 0.2 and boosts["conflict_penalty"] == 0.3
    assert set(boosts["sources"].values()) == {"deployed"}


@pytest.mark.asyncio
async def test_search_explicit_kwargs_beat_snapshot(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, {"deployed": {"usage_boost": 0.2, "confirmed_boost": 0.02, "conflict_penalty": 0.3}})
    _seed_edge_usage(tmp_path, [
        {"ts": "2026-11-18T01:00:00", "path": MIRROR, "outcome": "solved", "session_id": "s1"},
    ])
    step = _wire_search(monkeypatch, tmp_path, _search_step(tmp_path, usage_boost=0.4, confirmed_boost=0.1))
    response = await step(RuntimeContext(query="disk full"))

    top = response.metadata["results"][0]
    assert top["score"] == pytest.approx(0.5 + 0.4)  # explicit usage_boost wins
    boosts = response.metadata["ranking_boosts"]
    assert boosts["usage_boost"] == 0.4 and boosts["sources"]["usage_boost"] == "kwargs"
    assert boosts["confirmed_boost"] == 0.1 and boosts["sources"]["confirmed_boost"] == "kwargs"
    assert boosts["conflict_penalty"] == 0.3 and boosts["sources"]["conflict_penalty"] == "deployed"


@pytest.mark.asyncio
async def test_search_malformed_snapshot_falls_back_to_builtin(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, {})  # missing "deployed"
    _seed_edge_usage(tmp_path, [
        {"ts": "2026-11-18T01:00:00", "path": MIRROR, "outcome": "failed", "session_id": "s1"},
    ])
    step = _wire_search(monkeypatch, tmp_path, _search_step(tmp_path, conflict_penalty=0.5))
    response = await step(RuntimeContext(query="disk full"))

    boosts = response.metadata["ranking_boosts"]
    assert boosts["usage_boost"] == 0.05 and boosts["sources"]["usage_boost"] == "builtin"
    assert boosts["conflict_penalty"] == 0.5  # explicit still lands
    assert load_ranking_boosts(tmp_path, "metadata") == {}


# ----- config wiring -----------------------------------------------------------


def test_policy_anchor_carries_ranking_calibration():
    cloud = _load_config("ops_cloud.yaml")
    backends = [step["backend"] for step in cloud["jobs"]["ops_policy_evolve"]["steps"]]
    assert backends == ["ops_replay_step", "ops_policy_evolve_step", "ops_ranking_calibrate_step"]
    cron = cloud["jobs"]["ops_policy_evolve_cron"]
    assert [step["backend"] for step in cron["steps"]] == backends  # shared anchor
    edge = _load_config("ops_edge.yaml")
    search = edge["jobs"]["ops_search"]["steps"][0]
    assert "usage_boost" not in search  # explicit keys removed: snapshot must win
    assert "confirmed_boost" not in search and "conflict_penalty" not in search
