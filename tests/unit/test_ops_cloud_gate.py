"""Cloud retrieval gate: dimensionless planning criteria and two-phase search.

MATM (arXiv 2606.19911 §4.3) runs a RetrievalPlanner that decides
``[Retrieval]`` / ``[NoRetrieval]`` step by step, to avoid retrieval diluting
the context; its planner is explicitly replaceable by a deterministic
heuristic, which is what this gate is (no LLM). The gate only skips a cloud
round-trip — it never shrinks the candidate pool of a tier that was queried
(MATM §5.4: limiting the retrieval pool hurts), and it never weights a result
(ranking stays with usage boost and the verified ledger). Review finding 2:
absolute retrieval scores are never thresholds — they are non-normalized and
their dimension changes with the query and the single/dual-branch search mode.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.tiered_search import OpsSearchStep, should_call_cloud

MIRROR = "digest/muscle/disk-cleanup.md"
TOP = "digest/muscle/h0.md"  # the top hit path the evidence gate reads

THRESHOLD = {"min_solved": 3, "min_rate": 0.8, "dominance": 0.7}


def _hits(*scores: float) -> list[dict]:
    return [
        {"path": f"digest/muscle/h{i}.md", "score": score, "snippet": "s"}
        for i, score in enumerate(scores)
    ]


def _stats(solved: int, failed: int = 0, rate: float | None = None) -> dict:
    actual_rate = round(solved / (solved + failed), 4) if rate is None else rate
    return {TOP: {"solved": solved, "failed": failed, "success_rate": actual_rate}}


# ----- pure gate ---------------------------------------------------------------


def test_evidence_gate_skips_cloud_on_strong_reflex_evidence():
    stats = _stats(solved=5, failed=0, rate=1.0)
    assert should_call_cloud(_hits(0.5), stats, THRESHOLD) is False  # solved=5 >= 3, rate=1.0 >= 0.8
    # below the floor with two close hits: neither gate fires -> consult cloud
    assert should_call_cloud(_hits(0.5, 0.45), stats, {**THRESHOLD, "min_solved": 6}) is True


def test_evidence_gate_requires_both_floor_and_rate():
    low_rate = _stats(solved=5, failed=5)
    close = _hits(0.5, 0.45)  # ratio 0.526 < 0.7: dominance does not rescue
    assert should_call_cloud(close, low_rate, THRESHOLD) is True  # rate 0.5 < 0.8: falls through
    boundary = _stats(solved=3, failed=0)  # solved=3 >= 3, rate=0.8 >= 0.8
    assert should_call_cloud(close, boundary, THRESHOLD) is False  # >= semantics on both


def test_dominance_gate_is_the_thin_evidence_fallback():
    thin = _stats(solved=0, failed=0, rate=0.0)
    dominant = _hits(0.5, 0.1)  # top1/(top1+top2) = 0.833 >= 0.7
    assert should_call_cloud(dominant, thin, THRESHOLD) is False
    close = _hits(0.5, 0.45)  # 0.526 < 0.7: two close hits keep the cloud
    assert should_call_cloud(close, thin, THRESHOLD) is True
    single = _hits(0.5)  # one hit dominates by definition (ratio 1.0)
    assert should_call_cloud(single, thin, THRESHOLD) is False


def test_gate_disabled_or_empty_hits_always_call_cloud():
    assert should_call_cloud(_hits(0.5), _stats(solved=9, failed=0, rate=1.0), {"min_solved": 0}) is True
    assert should_call_cloud(_hits(0.5), {}, {**THRESHOLD, "min_solved": 0}) is True
    assert should_call_cloud([], _stats(solved=9, failed=0, rate=1.0), THRESHOLD) is True


def test_zero_scores_fall_back_to_calling_cloud():
    assert should_call_cloud(_hits(0.0, 0.0), _stats(solved=0, failed=1), THRESHOLD) is True


# ----- step: two-phase execution -----------------------------------------------


def _seed_edge_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stub_cloud(monkeypatch, calls: list[tuple[str, dict]] | None = None, fail: bool = False):
    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            if calls is not None:
                calls.append((action, payload))
            if fail:
                raise ConnectionError("connection refused")
            return {"success": True, "metadata": {"hits": [{"path": "digest/procedure/cloud.md", "name": "cloud", "description": "d", "score": 0.2}]}}

    monkeypatch.setattr("reme.steps.ops.tiered_search.CloudClient", StubCloud)


def _search_step(tmp_path: Path, **kwargs):
    return OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        cloud_host="127.0.0.1",
        cloud_port=2444,
        **kwargs,
    )


def _wire_search(monkeypatch, tmp_path: Path, step: OpsSearchStep, cloud_calls=None, cloud_fail=False):
    _stub_cloud(monkeypatch, cloud_calls, cloud_fail)

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = [{"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}}] if prefix == "digest/muscle" else []
        response = Mock()
        response.success = True
        response.metadata = {"results": hits}
        return response

    step.run_job = AsyncMock(side_effect=fake_run_job)
    return step


def _strong_usage(tmp_path: Path) -> None:
    _seed_edge_usage(
        tmp_path,
        [{"ts": f"2026-11-18T0{i}:00:00", "path": MIRROR, "outcome": "solved"} for i in range(5)],
    )


@pytest.mark.asyncio
async def test_gate_on_strong_evidence_skips_node_search(tmp_path, monkeypatch):
    _strong_usage(tmp_path)
    cloud_calls: list[tuple[str, dict]] = []
    step = _wire_search(
        monkeypatch,
        tmp_path,
        _search_step(tmp_path, cloud_gate_min_solved=3, cloud_gate_min_rate=0.8),
        cloud_calls=cloud_calls,
    )
    response = await step(RuntimeContext(query="disk full"))

    assert cloud_calls == []  # node_search never fired
    assert response.metadata["cloud_gate"] == {"min_solved": 3, "min_rate": 0.8, "dominance": 0.7, "triggered": True}
    assert response.metadata["tiers"]["cloud"] == {"hits": 0, "error": None, "skipped": True}
    assert [item["path"] for item in response.metadata["results"]] == [MIRROR]
    assert response.success is True


@pytest.mark.asyncio
async def test_gate_on_thin_evidence_falls_back_to_dominance_then_cloud(tmp_path, monkeypatch):
    # thin usage + the single muscle hit dominates the tier: gate fires
    cloud_calls: list[tuple[str, dict]] = []
    step = _wire_search(
        monkeypatch,
        tmp_path,
        _search_step(tmp_path, cloud_gate_min_solved=3),
        cloud_calls=cloud_calls,
    )
    response = await step(RuntimeContext(query="disk full"))
    assert cloud_calls == []  # single hit -> dominance 1.0 -> skip
    assert response.metadata["cloud_gate"]["triggered"] is True

    # thin usage + close evidence cannot be simulated with one dominant muscle
    # hit, so wire two close muscle hits (ratio 0.545 < 0.7): gate stays open
    cloud_calls2: list[tuple[str, dict]] = []
    step2 = _search_step(tmp_path, cloud_gate_min_solved=3)
    _stub_cloud(monkeypatch, cloud_calls2)

    async def fake_run_job_two(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hits = (
            [
                {"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}},
                {"path": "digest/muscle/alt.md", "text": "alt", "scores": {"score": 0.025}},
            ]
            if prefix == "digest/muscle"
            else []
        )
        response = Mock()
        response.success = True
        response.metadata = {"results": hits}
        return response

    step2.run_job = AsyncMock(side_effect=fake_run_job_two)
    response2 = await step2(RuntimeContext(query="disk full"))
    assert len(cloud_calls2) == 1 and cloud_calls2[0][0] == "node_search"  # cloud consulted
    assert response2.metadata["cloud_gate"]["triggered"] is False
    assert "skipped" not in response2.metadata["tiers"]["cloud"]
    assert [item["path"] for item in response2.metadata["results"]][:2] == [MIRROR, "digest/muscle/alt.md"]


@pytest.mark.asyncio
async def test_gate_off_is_the_concurrent_status_quo(tmp_path, monkeypatch):
    _strong_usage(tmp_path)  # evidence that WOULD have triggered the gate
    cloud_calls: list[tuple[str, dict]] = []
    step = _wire_search(monkeypatch, tmp_path, _search_step(tmp_path), cloud_calls=cloud_calls)
    response = await step(RuntimeContext(query="disk full"))

    assert len(cloud_calls) == 1 and cloud_calls[0][0] == "node_search"  # cloud still consulted
    assert "cloud_gate" not in response.metadata  # zero behaviour change when off
    assert response.metadata["tiers"]["cloud"] == {"hits": 1, "error": None}
    assert [item["path"] for item in response.metadata["results"]] == [MIRROR, "digest/procedure/cloud.md"]


@pytest.mark.asyncio
async def test_gate_triggered_survives_cloud_outage_semantics(tmp_path, monkeypatch):
    """When the cloud is consulted and unreachable, degradation is unchanged."""
    _seed_edge_usage(
        tmp_path,
        [{"ts": "2026-11-18T01:00:00", "path": MIRROR, "outcome": "failed"}],  # evidence fails the gate
    )

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        # two close muscle hits: dominance 0.6 < 0.7 -> the gate stays open
        hits = (
            [
                {"path": MIRROR, "text": "runbook", "scores": {"score": 0.03}},
                {"path": "digest/muscle/alt.md", "text": "alt", "scores": {"score": 0.025}},
            ]
            if prefix == "digest/muscle"
            else []
        )
        response = Mock()
        response.success = True
        response.metadata = {"results": hits}
        return response

    step = _search_step(tmp_path, cloud_gate_min_solved=3)
    _stub_cloud(monkeypatch, None, fail=True)
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    assert response.success is True
    assert "connection refused" in str(response.metadata["tiers"]["cloud"]["error"])  # same degradation text
    assert response.metadata["cloud_gate"]["triggered"] is False


@pytest.mark.asyncio
async def test_gate_triggered_never_calls_the_dead_cloud(tmp_path, monkeypatch):
    """Triggered gate: the outage is not even attempted, no error surfaces."""
    _strong_usage(tmp_path)
    step = _wire_search(
        monkeypatch,
        tmp_path,
        _search_step(tmp_path, cloud_gate_min_solved=3),
        cloud_fail=True,
    )
    response = await step(RuntimeContext(query="disk full"))

    assert response.success is True
    assert response.metadata["tiers"]["cloud"]["skipped"] is True
    assert response.metadata["tiers"]["cloud"]["error"] is None
    assert [item["path"] for item in response.metadata["results"]] == [MIRROR]


# ----- config wiring -----------------------------------------------------------


def test_edge_search_ships_the_gate_disabled():
    edge = _load_config("ops_edge.yaml")
    search = edge["jobs"]["ops_search"]["steps"][0]
    assert search["backend"] == "ops_search_step"
    assert search["cloud_gate_min_solved"] == 0  # default off: zero behaviour change
    assert search["infer_last_category"] is True  # earlier round intact
    assert search["preflight_warnings"] is True
