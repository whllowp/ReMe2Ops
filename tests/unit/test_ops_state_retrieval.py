"""State-conditioned retrieval: infer last_category from session usage history.

MATM (arXiv 2606.19911 §3.1): the retrieval key is the task description plus
the recent trajectory — a consumer searching by its current state finds more
than one searching by the original task. Here the trajectory is the session's
local usage history; its newest hit's category becomes the transition key, so
``transition_boost`` works without extra wiring. Caller-supplied
``last_category`` always wins; no history means exactly the old behaviour.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.transitions import category_of, save_graphs
from reme.steps.ops.tiered_search import OpsSearchStep, infer_last_category

MIRROR = "digest/muscle/disk-cleanup.md"

RECORDS = [
    {"ts": "2026-11-18T01:00:00", "path": "digest/muscle/disk-cleanup.md", "session_id": "s1", "outcome": "solved"},
    {"ts": "2026-11-18T02:00:00", "path": "digest/muscle/disk-alert.md", "session_id": "s1", "outcome": "solved"},
    {"ts": "2026-11-18T03:00:00", "path": "digest/muscle/log-rotate.md", "session_id": "s2", "outcome": "failed"},
    {"ts": "2026-11-18T04:00:00", "path": "digest/muscle/disk-alert.md", "outcome": "solved"},  # no session
    "junk",
]


# ----- pure function ----------------------------------------------------------


def test_infer_last_category_prefers_latest_session_hit():
    assert infer_last_category(RECORDS, "s1") == "disk-alert"  # ts 02:00 beats 01:00
    assert infer_last_category(RECORDS, "s2") == "log-rotate"


def test_infer_last_category_empty_session_falls_back_to_global():
    # no session filter: the newest record overall wins (ts 04:00)
    assert infer_last_category(RECORDS) == "disk-alert"


def test_infer_last_category_ties_break_to_later_record_position():
    tied = [
        {"ts": "t1", "path": "digest/muscle/first.md", "session_id": "s1"},
        {"ts": "t1", "path": "digest/muscle/second.md", "session_id": "s1"},
    ]
    assert infer_last_category(tied, "s1") == "second"


def test_infer_last_category_no_history_yields_empty():
    assert infer_last_category([], "s1") == ""
    assert infer_last_category([{"ts": "t1", "path": "digest/muscle/a.md", "session_id": "s9"}], "s1") == ""
    assert infer_last_category([{"path": "digest/muscle/a.md"}, "junk"], "") == "a"


# ----- step wiring ------------------------------------------------------------


def _seed_edge_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _seed_graphs(tmp_path: Path) -> None:
    save_graphs(tmp_path, "metadata", {"global": {"disk-alert->disk-cleanup": 4.0}})


def _search_step(tmp_path: Path, **kwargs):
    return OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        cloud_host="127.0.0.1",
        cloud_port=2444,
        **kwargs,
    )


def _wire(monkeypatch, tmp_path: Path, step: OpsSearchStep) -> OpsSearchStep:
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
async def test_session_history_infers_category_and_triggers_support(tmp_path, monkeypatch):
    _seed_edge_usage(
        tmp_path,
        [
            {"ts": "2026-11-18T01:00:00", "path": "digest/muscle/disk-cleanup.md", "session_id": "s1", "outcome": "solved"},
            {"ts": "2026-11-18T02:00:00", "path": "digest/muscle/disk-alert.md", "session_id": "s1", "outcome": "solved"},
        ],
    )
    _seed_graphs(tmp_path)
    step = _wire(
        monkeypatch,
        tmp_path,
        _search_step(tmp_path, transition_boost=0.1, infer_last_category=True, usage_boost=0.0),
    )
    response = await step(RuntimeContext(query="disk full", session_id="s1"))

    top = response.metadata["results"][0]
    assert top["transition_support"] == pytest.approx(1.0)  # global view: the only edge fires
    assert top["score"] == pytest.approx(0.5 + 0.1)  # rank-1 muscle weight + 0.1 * support
    assert response.metadata["inferred_last_category"] == "disk-alert"  # session's newest hit


@pytest.mark.asyncio
async def test_no_session_history_behaves_as_before(tmp_path, monkeypatch):
    _seed_edge_usage(
        tmp_path,
        [{"ts": "2026-11-18T01:00:00", "path": "digest/muscle/disk-alert.md", "session_id": "s9", "outcome": "solved"}],
    )
    _seed_graphs(tmp_path)
    step = _wire(monkeypatch, tmp_path, _search_step(tmp_path, transition_boost=0.1, infer_last_category=True))
    response = await step(RuntimeContext(query="disk full", session_id="s1"))  # s1 has no history

    top = response.metadata["results"][0]
    assert top["score"] == pytest.approx(0.5)  # no support, no boost
    assert "transition_support" not in top
    assert "inferred_last_category" not in response.metadata


@pytest.mark.asyncio
async def test_explicit_last_category_beats_inference(tmp_path, monkeypatch):
    _seed_edge_usage(
        tmp_path,
        [{"ts": "2026-11-18T01:00:00", "path": "digest/muscle/disk-alert.md", "session_id": "s1", "outcome": "solved"}],
    )
    _seed_graphs(tmp_path)
    # caller pins an unrelated category: inference must not override it
    step = _wire(monkeypatch, tmp_path, _search_step(tmp_path, transition_boost=0.1, infer_last_category=True))
    response = await step(RuntimeContext(query="disk full", session_id="s1", last_category="log-rotate"))

    top = response.metadata["results"][0]
    assert top["score"] == pytest.approx(0.5)  # log-rotate has no outgoing edge
    assert "transition_support" not in top
    assert "inferred_last_category" not in response.metadata

    # and without the kwarg an explicit key still drives support (existing path)
    step2 = _wire(monkeypatch, tmp_path, _search_step(tmp_path, transition_boost=0.1))
    response2 = await step2(RuntimeContext(query="disk full", last_category="disk-alert"))
    top2 = response2.metadata["results"][0]
    assert top2["score"] == pytest.approx(0.5 + 0.1)
    assert top2["transition_support"] == pytest.approx(1.0)
    assert "inferred_last_category" not in response2.metadata


@pytest.mark.asyncio
async def test_inference_without_boost_never_touches_the_answer(tmp_path, monkeypatch):
    """transition_boost=0 leaves the response identical to the legacy path."""
    _seed_edge_usage(
        tmp_path,
        [{"ts": "2026-11-18T01:00:00", "path": "digest/muscle/disk-alert.md", "session_id": "s1", "outcome": "solved"}],
    )
    step = _wire(monkeypatch, tmp_path, _search_step(tmp_path, infer_last_category=True))
    response = await step(RuntimeContext(query="disk full", session_id="s1"))

    assert response.metadata["results"][0]["score"] == pytest.approx(0.5)
    assert "inferred_last_category" not in response.metadata


# ----- config wiring ----------------------------------------------------------


def test_edge_search_carries_infer_switch():
    edge = _load_config("ops_edge.yaml")
    search = edge["jobs"]["ops_search"]["steps"][0]
    assert search["backend"] == "ops_search_step"
    assert search["infer_last_category"] is True
    # the second half of the wiring: without transition_boost > 0 the support
    # term stays inert and the inference switch is dead config
    assert float(search["transition_boost"]) > 0.0
    assert category_of(MIRROR) == "disk-cleanup"  # shared category rule still holds
