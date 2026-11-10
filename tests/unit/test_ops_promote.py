"""Ops muscle promotion: threshold scan, fast track, and the promote step contract."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.as_decision_model.base_as_decision_model import BaseAsDecisionModel
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.promote import OpsMusclePromoteStep, OpsPromotionScanStep, promotion_candidates

SOP = "digest/procedure/disk-space-cleanup.md"


class FakeDecision(BaseAsDecisionModel):
    """Callable decision model returning a fixed probability for every question."""

    def __init__(self, probability: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.calls = []

    async def __call__(self, state, questions):
        self.calls.append((state, questions))
        content = {name: SimpleNamespace(type="binary_answer", probability=self.probability) for name in questions}
        return SimpleNamespace(content=content, usage=None, model="fake")


def _store(tmp_path):
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    return store


def test_promotion_candidates_filters_and_orders():
    usage_stats = {
        "digest/procedure/good.md": {"solved": 9, "failed": 1, "success_rate": 0.9},
        "digest/procedure/best.md": {"solved": 12, "failed": 0, "success_rate": 1.0},
        "digest/procedure/too-few.md": {"solved": 2, "failed": 0, "success_rate": 1.0},
        "digest/procedure/too-flaky.md": {"solved": 8, "failed": 4, "success_rate": 0.6667},
        "digest/muscle/already.md": {"solved": 20, "failed": 0, "success_rate": 1.0},
        "daily/2026-11-06/n.md": {"solved": 30, "failed": 0, "success_rate": 1.0},
    }
    candidates = promotion_candidates(
        usage_stats,
        procedure_dir="digest/procedure",
        min_solved=5,
        min_success_rate=0.8,
    )
    assert [path for path, _ in candidates] == ["digest/procedure/best.md", "digest/procedure/good.md"]
    assert candidates[0][1]["solved"] == 12


def _step(tmp_path, **kwargs):
    return OpsMusclePromoteStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        **kwargs,
    )


def _seed_usage(tmp_path, records: list[dict]):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.mark.asyncio
async def test_promote_step_moves_and_stamps_frontmatter(tmp_path):
    source = tmp_path / "digest" / "procedure" / "disk-full.md"
    source.parent.mkdir(parents=True)
    source.write_text("---\nname: disk-full\n---\nRunbook body\n", encoding="utf-8")

    step = _step(tmp_path)
    step.run_job = AsyncMock(
        side_effect=[
            Mock(success=True, answer="Moved"),
            Mock(success=True, answer="stamped"),
        ],
    )
    response = await step(RuntimeContext(path="digest/procedure/disk-full.md"))

    assert response.success is True
    assert response.metadata["promoted"] == {"from": "digest/procedure/disk-full.md", "to": "digest/muscle/disk-full.md"}
    move_call = step.run_job.await_args_list[0]
    assert move_call.args[0] == "move"
    assert move_call.kwargs == {
        "src_path": "digest/procedure/disk-full.md",
        "dst_path": "digest/muscle/disk-full.md",
        "overwrite": True,
    }
    stamp_call = step.run_job.await_args_list[1]
    assert stamp_call.args[0] == "frontmatter_update"
    assert stamp_call.kwargs["path"] == "digest/muscle/disk-full.md"
    assert stamp_call.kwargs["metadata"]["memory_type"] == "muscle"
    assert "promoted_by" not in stamp_call.kwargs["metadata"]
    # downstream reflex-card step consumes this context entry
    assert step.context["promoted"] == {"from": "digest/procedure/disk-full.md", "to": "digest/muscle/disk-full.md"}


@pytest.mark.asyncio
async def test_promote_step_stamps_decision_model_verdict(tmp_path):
    source = tmp_path / SOP
    source.parent.mkdir(parents=True)
    source.write_text("body", encoding="utf-8")
    step = _step(tmp_path)
    step.run_job = AsyncMock(
        side_effect=[
            Mock(success=True, answer="Moved"),
            Mock(success=True, answer="stamped"),
        ],
    )
    response = await step(RuntimeContext(path=SOP, judge_probability=0.93))

    assert response.success is True
    stamp_call = step.run_job.await_args_list[1]
    metadata = stamp_call.kwargs["metadata"]
    assert metadata["promoted_by"] == "decision-model"
    assert metadata["judge_probability"] == pytest.approx(0.93)


@pytest.mark.parametrize(
    "path,expected_error",
    [
        ("", "path is required"),
        ("daily/2026-11-06/n.md", "only long-term digest"),
        ("digest/muscle/a.md", "only long-term digest"),
        ("digest/procedure/missing.md", "does not exist"),
    ],
    ids=["empty", "daily", "muscle", "missing"],
)
@pytest.mark.asyncio
async def test_promote_step_validations(tmp_path, path, expected_error):
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    step = _step(tmp_path)
    step.run_job = AsyncMock()
    response = await step(RuntimeContext(path=path))
    assert response.success is False
    assert expected_error in response.answer
    step.run_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_promote_step_propagates_move_failure(tmp_path):
    source = tmp_path / "digest" / "procedure" / "a.md"
    source.parent.mkdir(parents=True)
    source.write_text("body", encoding="utf-8")
    step = _step(tmp_path)
    step.run_job = AsyncMock(return_value=Mock(success=False, answer="dst exists"))
    response = await step(RuntimeContext(path="digest/procedure/a.md"))
    assert response.success is False
    assert "dst exists" in response.answer
    assert step.run_job.await_count == 1


@pytest.mark.asyncio
async def test_scan_fast_track_promotes_confident_low_usage_sop(tmp_path):
    """A SOP with usage below the thresholds is promoted on a confident verdict."""
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    source = tmp_path / SOP
    source.write_text("runbook body", encoding="utf-8")
    _seed_usage(tmp_path, [
        {"ts": "t1", "path": SOP, "outcome": "solved"},
        {"ts": "t2", "path": SOP, "outcome": "solved"},
    ])  # solved=2, rate=1.0: below min_solved=5 -> only the fast track can take it

    step = OpsPromotionScanStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(probability=0.9),
        decision_fast_track=True,
        fast_track_min_solved=1,
        muscle_worthy_threshold=0.8,
        evidence_threshold=0.6,
    )
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))
    response = await step(RuntimeContext())

    meta = response.metadata["promotion"]
    assert [entry["path"] for entry in meta["fast_tracked"]] == [SOP]
    promote_kwargs = step.run_job.await_args.kwargs
    assert promote_kwargs["judge_probability"] == pytest.approx(0.9)
    assert "decision model, 1 promoted" in response.answer


@pytest.mark.asyncio
async def test_scan_fast_track_skips_low_confidence(tmp_path):
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    source = tmp_path / SOP
    source.write_text("runbook body", encoding="utf-8")
    _seed_usage(tmp_path, [{"ts": "t1", "path": SOP, "outcome": "solved"}])

    step = OpsPromotionScanStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(probability=0.2),
        decision_fast_track=True,
    )
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))
    response = await step(RuntimeContext())

    assert response.metadata["promotion"]["fast_tracked"] == []
    step.run_job.assert_not_awaited()
