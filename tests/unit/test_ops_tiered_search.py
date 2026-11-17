"""Tiered ops search: per-path collapse, tier merge, usage boost, status weighting."""

import json
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileFrontMatter
from reme.steps.ops.dream_audit import body_sha256
from reme.steps.ops.tiered_search import OpsSearchStep, best_path_scores, combined_usage, merge_tier_hits


def test_best_path_scores_keeps_best_chunk_per_path():
    results = [
        {"path": "digest/muscle/a.md", "text": "weak", "scores": {"score": 0.01}},
        {"path": "digest/muscle/a.md", "text": "strong", "scores": {"score": 0.03}},
        {"path": "daily/2026-11-06/n.md", "text": "daily", "scores": {"score": 0.02}},
        {"path": "", "text": "no path", "scores": {"score": 0.5}},
    ]
    best = best_path_scores(results)
    assert set(best) == {"digest/muscle/a.md", "daily/2026-11-06/n.md"}
    assert best["digest/muscle/a.md"]["snippet"] == "strong"
    assert best["digest/muscle/a.md"]["score"] == 0.03


def test_merge_tier_hits_weights_dedups_and_boosts():
    tiers = [
        {"tier": "muscle", "weight": 1.0, "hits": [{"path": "digest/muscle/a.md", "snippet": "runbook"}]},
        {
            "tier": "daily",
            "weight": 0.6,
            "hits": [
                {"path": "daily/2026-11-06/incident.md", "snippet": "log"},
                {"path": "digest/muscle/a.md", "snippet": "dup"},
            ],
        },
        {"tier": "cloud", "weight": 0.4, "hits": [{"path": "digest/procedure/z.md", "name": "z-sop"}]},
    ]
    usage_stats = {"digest/muscle/a.md": {"solved": 10, "failed": 0}}
    merged = merge_tier_hits(tiers, limit=5, usage_stats=usage_stats, usage_boost=0.05)

    paths = [item["path"] for item in merged]
    assert paths[0] == "digest/muscle/a.md"  # two tier contributions + usage boost
    top = merged[0]
    assert top["tiers"] == ["muscle", "daily"]
    assert top["score"] == pytest.approx(1.0 / 2 + 0.6 / 3 + 0.05 * 10)
    assert top["snippet"] == "runbook"  # first-seen metadata wins
    assert merged[2]["name"] == "z-sop"


def test_merge_tier_hits_respects_limit():
    tiers = [
        {
            "tier": "daily",
            "weight": 1.0,
            "hits": [{"path": f"daily/n{i}.md", "snippet": ""} for i in range(8)],
        },
    ]
    merged = merge_tier_hits(tiers, limit=3, usage_stats={}, usage_boost=0.05)
    assert len(merged) == 3


def test_merge_tier_hits_without_boost():
    tiers = [{"tier": "muscle", "weight": 1.0, "hits": [{"path": "digest/muscle/a.md"}]}]
    merged = merge_tier_hits(tiers, limit=5, usage_stats={}, usage_boost=0.05)
    assert "usage_boosted" not in merged[0]


def test_merge_tier_hits_status_weighting_moves_ranking():
    """Equal-score muscle hits: confirmed rises, conflict sinks, neutral is unchanged."""
    tiers = [
        {
            "tier": "muscle",
            "weight": 1.0,
            "hits": [{"path": "digest/muscle/a.md"}, {"path": "digest/muscle/b.md"}],
        },
        {
            "tier": "daily",
            "weight": 1.0,
            "hits": [{"path": "digest/muscle/b.md"}, {"path": "digest/muscle/a.md"}],
        },
    ]
    base = 1 / 2 + 1 / 3  # a and b accumulate identical scores across the tiers

    confirmed = merge_tier_hits(
        tiers,
        limit=5,
        usage_stats={},
        usage_boost=0.05,
        status_by_path={"digest/muscle/a.md": "confirmed"},
        confirmed_boost=0.05,
    )
    assert [item["path"] for item in confirmed] == ["digest/muscle/a.md", "digest/muscle/b.md"]
    assert confirmed[0]["score"] == pytest.approx(base + 0.05)
    assert confirmed[0]["verified_status"] == "confirmed"
    assert "verified_status" not in confirmed[1]

    conflict = merge_tier_hits(
        tiers,
        limit=5,
        usage_stats={},
        usage_boost=0.05,
        status_by_path={"digest/muscle/a.md": "conflict"},
        conflict_penalty=0.15,
    )
    assert [item["path"] for item in conflict] == ["digest/muscle/b.md", "digest/muscle/a.md"]
    assert conflict[1]["score"] == pytest.approx(base - 0.15)

    neutral = merge_tier_hits(tiers, limit=5, usage_stats={}, usage_boost=0.05)
    assert neutral[0]["score"] == pytest.approx(neutral[1]["score"])  # no status: identical
    assert all("verified_status" not in item for item in neutral)


def test_combined_usage_sums_promotion_alias():
    usage_stats = {
        "digest/procedure/a.md": {"solved": 5, "failed": 1},
        "digest/muscle/a.md": {"solved": 3, "failed": 0},
    }
    usage = combined_usage("digest/muscle/a.md", usage_stats, {"digest/muscle/a.md": "digest/procedure/a.md"})
    assert usage == {"solved": 8, "failed": 1}
    assert combined_usage("digest/muscle/b.md", usage_stats, {"digest/muscle/a.md": "digest/procedure/a.md"}) == {
        "solved": 0,
        "failed": 0,
    }


def test_merge_tier_hits_boosts_through_promotion_alias():
    tiers = [{"tier": "muscle", "weight": 1.0, "hits": [{"path": "digest/muscle/a.md"}]}]
    merged = merge_tier_hits(
        tiers,
        limit=5,
        usage_stats={"digest/procedure/a.md": {"solved": 9, "failed": 1}},
        usage_boost=0.05,
        usage_aliases={"digest/muscle/a.md": "digest/procedure/a.md"},
    )
    assert merged[0]["usage"] == {"solved": 9, "failed": 1}
    assert merged[0]["score"] == pytest.approx(0.5 + 0.45)
    assert merged[0]["usage_boosted"] is True


@pytest.mark.asyncio
async def test_ops_search_step_resolves_promotion_aliases(tmp_path, monkeypatch):
    """Muscle hits inherit usage history recorded under their pre-promotion path."""
    muscle_results = [{"path": "digest/muscle/a.md", "text": "runbook", "scores": {"score": 0.03}}]

    async def fake_run_job(name, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        return _search_response(muscle_results if prefix == "digest/muscle" else [])

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
    node = Mock(path="digest/muscle/a.md", front_matter=FileFrontMatter(name="a", promoted_from="digest/procedure/a.md"))
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    store.get_nodes = AsyncMock(return_value=[node])
    usage_file = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    usage_file.parent.mkdir(parents=True)
    for i in range(9):
        with usage_file.open("a", encoding="utf-8") as fh:
            fh.write(f'{{"ts": "t{i}", "path": "digest/procedure/a.md", "outcome": "solved"}}\n')

    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
        cloud_host="127.0.0.1",
        cloud_port=2444,
    )
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    top = response.metadata["results"][0]
    assert top["usage"]["solved"] == 9
    assert top["usage_boosted"] is True
    store.get_nodes.assert_awaited_once_with(["digest/muscle/a.md"])


def _search_response(results):
    response = Mock()
    response.success = True
    response.metadata = {"results": results}
    return response


@pytest.mark.asyncio
async def test_ops_search_step_merges_local_and_cloud(tmp_path, monkeypatch):
    muscle_results = [{"path": "digest/muscle/a.md", "text": "runbook body", "scores": {"score": 0.03}}]
    daily_results = [{"path": "daily/2026-11-06/incident.md", "text": "incident log", "scores": {"score": 0.02}}]
    cloud_payload = {
        "success": True,
        "metadata": {"hits": [{"path": "digest/procedure/z.md", "name": "z", "description": "d", "score": 0.5}]},
    }

    async def fake_run_job(name, **kwargs):
        assert name == "search"
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        return _search_response(muscle_results if prefix == "digest/muscle" else daily_results)

    class StubCloud:
        def __init__(self, host, port, timeout=8.0, transport=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            assert action == "node_search"
            return cloud_payload

    monkeypatch.setattr("reme.steps.ops.tiered_search.CloudClient", StubCloud)
    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        tier_limit=5,
        cloud_host="127.0.0.1",
        cloud_port=2444,
    )
    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(query="disk full"))

    assert response.success is True
    paths = [item["path"] for item in response.metadata["results"]]
    assert paths == ["digest/muscle/a.md", "daily/2026-11-06/incident.md", "digest/procedure/z.md"]
    tiers = response.metadata["tiers"]
    assert tiers["muscle"]["hits"] == 1 and tiers["daily"]["hits"] == 1 and tiers["cloud"]["hits"] == 1
    filters = [call.kwargs["search_filter"] for call in step.run_job.await_args_list]
    assert filters == [
        {"path_prefixes": ["digest/muscle"]},
        {"path_prefixes": ["daily"]},
    ]


@pytest.mark.asyncio
async def test_ops_search_step_survives_cloud_outage(tmp_path, monkeypatch):
    class StubCloud:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setattr("reme.steps.ops.tiered_search.CloudClient", StubCloud)
    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        cloud_host="127.0.0.1",
        cloud_port=2444,
    )
    step.run_job = AsyncMock(
        return_value=_search_response([{"path": "digest/muscle/a.md", "text": "b", "scores": {"score": 0.03}}]),
    )
    context = RuntimeContext(query="disk full")
    response = await step(context)

    assert response.success is True
    assert [item["path"] for item in response.metadata["results"]] == ["digest/muscle/a.md"]
    assert "connection refused" in str(response.metadata["tiers"]["cloud"]["error"])


@pytest.mark.asyncio
async def test_ops_search_step_requires_query(tmp_path, monkeypatch):
    class Unused:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cloud must not be reached")

    monkeypatch.setattr("reme.steps.ops.tiered_search.CloudClient", Unused)
    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )
    response = await step(RuntimeContext(query="   "))
    assert response.success is False
    assert "query" in response.answer


@pytest.mark.asyncio
async def test_ops_search_step_weights_from_hash_gated_ledger(tmp_path, monkeypatch):
    """Muscle status weighting comes from the ledger and only while the mirror
    body hash still matches; a tampered mirror drops back to neutral."""
    mirror = tmp_path / "digest" / "muscle" / "a.md"
    mirror.parent.mkdir(parents=True)
    stamp = {"name": "a", "promoted_from": "digest/procedure/a.md", "verified_status": "confirmed"}
    mirror.write_text(frontmatter.dumps(frontmatter.Post("ledger body v1", **stamp)), encoding="utf-8")
    ledger_path = tmp_path / "metadata" / "ops_audit" / "status.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"digest/muscle/a.md": {"status": "confirmed", "body_sha256": body_sha256("ledger body v1")}}),
        encoding="utf-8",
    )

    async def fake_run_job(name, /, **kwargs):
        prefix = kwargs["search_filter"]["path_prefixes"][0]
        hit = {"path": "digest/muscle/a.md", "text": "runbook", "scores": {"score": 0.03}}
        results = [hit] if prefix == "digest/muscle" else []
        return _search_response(results)

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
    node = Mock(
        path="digest/muscle/a.md",
        front_matter=FileFrontMatter(name="a", promoted_from="digest/procedure/a.md", verified_status="confirmed"),
    )
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    store.get_nodes = AsyncMock(return_value=[node])
    step = OpsSearchStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
        confirmed_boost=0.05,
        conflict_penalty=0.15,
    )
    step.run_job = AsyncMock(side_effect=fake_run_job)

    first = await step(RuntimeContext(query="disk full"))
    top = first.metadata["results"][0]
    assert top["verified_status"] == "confirmed"
    assert top["score"] == pytest.approx(0.5 + 0.05)  # muscle rank-1 weight + confirmed boost
    assert first.metadata["frontmatter_status"] == {"digest/muscle/a.md": "confirmed"}

    # Tampering the mirror body breaks the ledger hash: weighting zeroes out.
    mirror.write_text(frontmatter.dumps(frontmatter.Post("tampered body", **stamp)), encoding="utf-8")
    second = await step(RuntimeContext(query="disk full"))
    top2 = second.metadata["results"][0]
    assert "verified_status" not in top2
    assert top2["score"] == pytest.approx(0.5)  # neutral muscle weight only

    assert store.get_nodes.await_count == 2  # one frontmatter pass per search
    assert all(call.args[0] == ["digest/muscle/a.md"] for call in store.get_nodes.await_args_list)
