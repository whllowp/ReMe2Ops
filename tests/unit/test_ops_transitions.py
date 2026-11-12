"""Time-aware transition graphs and the tiered-search support bonus."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.transitions import (
    OpsTransitionsStep,
    build_transition_graphs,
    category_of,
    classify_time,
    hour_bin_name,
    transition_support,
    week_type_name,
)


def test_hour_bins_and_week_types():
    assert hour_bin_name(3) == "late_night"
    assert hour_bin_name(7) == "morning"
    assert hour_bin_name(23) == "night"
    assert week_type_name(0) == "weekday"  # Monday
    assert week_type_name(5) == "weekend"  # Saturday


def test_classify_time_yields_four_views():
    keys = classify_time("2026-11-11T08:30:00")  # Wednesday morning
    assert keys["global"] == "global"
    assert keys["week_type"] == "wt:weekday"
    assert keys["hour_bin"] == "bin:morning"
    assert keys["joint"] == "wt:weekday+bin:morning"


def test_build_graphs_counts_conditioned_transitions():
    records = [
        {"ts": "2026-11-11T08:00:00", "path": "digest/muscle/disk-alert.md", "source_device": "d1",
         "outcome": "solved"},
        {"ts": "2026-11-11T08:30:00", "path": "digest/muscle/disk-cleanup.md", "source_device": "d1",
         "outcome": "solved"},
    ]
    views = build_transition_graphs(records)
    edge = "disk-alert->disk-cleanup"
    assert views["global"][edge] == 1
    assert views["wt:weekday"][edge] == 1
    assert views["bin:morning"][edge] == 1
    assert views["wt:weekday+bin:morning"][edge] == 1


def test_transition_support_prefers_conditioned_view():
    graphs = {
        "global": {"a->b": 1.0, "a->c": 3.0},
        "wt:weekday": {"a->b": 3.0, "a->c": 1.0},
    }
    # weekday-morning replay: conditioned views dominate the global smoothing.
    support_b = transition_support(graphs, "a", "2026-11-11T08:00:00", "b")
    support_c = transition_support(graphs, "a", "2026-11-11T08:00:00", "c")
    assert support_b > support_c
    assert transition_support(graphs, "a", "2026-11-11T08:00:00", "missing") == 0.0
    assert category_of("digest/muscle/disk-cleanup.md") == "disk-cleanup"


@pytest.mark.asyncio
async def test_transitions_step_persists_graphs(tmp_path):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True)
    with usage.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-11-11T08:00:00", "path": "digest/muscle/a.md",
                             "outcome": "solved", "source_device": "d1"}) + "\n")
        fh.write(json.dumps({"ts": "2026-11-11T08:20:00", "path": "digest/muscle/b.md",
                             "outcome": "solved", "source_device": "d1"}) + "\n")
    step = OpsTransitionsStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)),
                              file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path))
    response = await step(RuntimeContext())
    assert response.success is True
    assert "bin:morning" in response.metadata["transitions"]["views"]
    graphs = json.loads((tmp_path / "metadata" / "ops_transitions" / "graphs.json").read_text(encoding="utf-8"))
    assert "views" in graphs


@pytest.mark.asyncio
async def test_search_applies_transition_boost(tmp_path, monkeypatch):
    """A matching time-conditioned transition lifts a hit above a stronger raw hit."""
    from reme.components.runtime_context import RuntimeContext as RC
    from reme.steps.ops.tiered_search import OpsSearchStep

    graphs = {
        "global": {"prev->hot-b.md": 1.0},
        "wt:weekday": {"prev->hot-b.md": 1.0},
        "bin:morning": {"prev->hot-b.md": 1.0},
        "wt:weekday+bin:morning": {"prev->hot-b.md": 1.0},
    }
    (tmp_path / "metadata" / "ops_transitions").mkdir(parents=True)
    (tmp_path / "metadata" / "ops_transitions" / "graphs.json").write_text(
        json.dumps({"built_at": "t", "views": graphs}), encoding="utf-8")

    async def fake_run_job(name, **kwargs):
        response = Mock()
        response.success = True
        path = "digest/muscle/hot-b.md" if kwargs["search_filter"]["path_prefixes"][0].endswith("muscle") else "daily/z.md"
        response.metadata = {"results": [{"path": path, "text": "body", "scores": {"score": 0.05}}]}
        return response

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
    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        transition_boost=0.5,
        last_category="prev",
        cloud_host="127.0.0.1",
        cloud_port=2444,
    )
    step.run_job = AsyncMock(side_effect=fake_run_job)

    # Deterministic structural support: only the transition-matching SOP earns it.
    import reme.steps.ops.tiered_search as ts

    monkeypatch.setattr(ts, "transition_support", lambda g, frm, at, tgt: (0.5 if tgt == "hot-b" else 0.0))
    response = await step(RC(query="anything", last_category="prev"))

    results = response.metadata["results"]
    assert results, "results present"
    boosted = [r for r in results if r.get("transition_support")]
    assert boosted and boosted[0]["path"] == "digest/muscle/hot-b.md"
    assert results[0]["path"] == "digest/muscle/hot-b.md"  # bonus flips the ranking
