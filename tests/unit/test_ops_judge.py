"""Ops decision-model fast track: judge_sops, the judge step, and ingest candidates.

Mocks are passed as constructor kwargs: ``BaseStep.__call__`` clears cached
``_ref_*`` entries on every invocation, so instance-attribute injection would
be discarded before ``execute`` runs.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.as_decision_model.base_as_decision_model import BaseAsDecisionModel
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.judge import OpsMuscleJudgeStep, judge_sops
from reme.steps.ops.usage_ingest import OpsUsageIngestStep, fast_track_candidates

SOP = "digest/procedure/disk-space-cleanup.md"


class FakeDecision(BaseAsDecisionModel):
    """Callable decision model returning a fixed probability for every question."""

    def __init__(self, probability: float = 0.95, error: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.error = error
        self.calls = []

    async def __call__(self, state, questions):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        content = {
            name: SimpleNamespace(type="binary_answer", probability=self.probability) for name in questions
        }
        return SimpleNamespace(content=content, usage=None, model="fake")


def _seed_sop(tmp_path, *, solved=2, failed=0):
    path = tmp_path / SOP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: disk-full\n---\nRunbook body\n", encoding="utf-8")
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for i in range(solved):
            fh.write(json.dumps({"ts": f"t{i}", "path": SOP, "outcome": "solved"}) + "\n")
        for i in range(failed):
            fh.write(json.dumps({"ts": f"f{i}", "path": SOP, "outcome": "failed"}) + "\n")


def _store(tmp_path):
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    return store


@pytest.mark.asyncio
async def test_judge_sops_confident_verdict_is_promotable(tmp_path):
    _seed_sop(tmp_path, solved=2)
    decision = FakeDecision(probability=0.92)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=decision,
    )

    verdicts = await judge_sops(step, [SOP])
    verdict = verdicts[SOP]
    assert verdict["judged"] is True
    assert verdict["muscle_worthy"] == pytest.approx(0.92)
    assert verdict["solved"] == 2
    assert verdict["promotable"] is True
    state, questions = decision.calls[0]
    candidate = state["candidates"][0]
    assert candidate["path"] == SOP
    assert candidate["usage"]["solved"] == 2
    assert "Runbook body" in candidate["body"]
    assert any(name.endswith("_muscle_worthy") for name in questions)
    assert any(name.endswith("_evidence_sufficient") for name in questions)


@pytest.mark.asyncio
async def test_judge_sops_low_confidence_not_promotable(tmp_path):
    _seed_sop(tmp_path, solved=2)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(probability=0.3),
    )

    verdicts = await judge_sops(step, [SOP])
    assert verdicts[SOP]["judged"] is True
    assert verdicts[SOP]["promotable"] is False


@pytest.mark.asyncio
async def test_judge_sops_min_solved_gates(tmp_path):
    _seed_sop(tmp_path, solved=0)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(probability=0.99),
    )

    verdicts = await judge_sops(step, [SOP])
    assert verdicts[SOP]["promotable"] is False  # confident but never used


@pytest.mark.asyncio
async def test_judge_sops_is_failure_safe(tmp_path):
    _seed_sop(tmp_path)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(error=ConnectionError("model down")),
    )

    verdicts = await judge_sops(step, [SOP, "digest/procedure/missing.md"])
    assert verdicts[SOP]["judged"] is False
    assert "model down" in verdicts[SOP]["reason"]
    assert verdicts["digest/procedure/missing.md"]["reason"] == "file missing"


@pytest.mark.asyncio
async def test_judge_sops_without_decision_model(tmp_path):
    _seed_sop(tmp_path)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
    )

    verdicts = await judge_sops(step, [SOP])
    assert verdicts[SOP]["judged"] is False
    assert "no decision model" in verdicts[SOP]["reason"]


@pytest.mark.asyncio
async def test_judge_step_auto_promotes(tmp_path):
    _seed_sop(tmp_path, solved=2)
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
        as_decision_model=FakeDecision(probability=0.9),
    )
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))

    response = await step(RuntimeContext(fast_track_candidates=[SOP]))

    assert response.success is True
    assert response.metadata["judge"]["promoted"] == [{"path": SOP, "muscle_worthy": pytest.approx(0.9)}]
    step.run_job.assert_awaited_once()
    assert step.run_job.await_args.kwargs["judge_probability"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_judge_step_without_candidates_is_noop(tmp_path):
    step = OpsMuscleJudgeStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
    )
    response = await step(RuntimeContext())
    assert response.metadata["judge"]["paths"] == []


def test_fast_track_candidates_filters(tmp_path):
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    (tmp_path / "digest" / "muscle").mkdir(parents=True)
    (tmp_path / "digest" / "procedure" / "a.md").write_text("body", encoding="utf-8")
    (tmp_path / "digest" / "muscle" / "m.md").write_text("body", encoding="utf-8")
    records = [
        {"path": "digest/procedure/a.md", "outcome": "solved"},
        {"path": "digest/procedure/a.md", "outcome": "failed"},
        {"path": "digest/muscle/m.md", "outcome": "solved"},
        {"path": "digest/procedure/missing.md", "outcome": "solved"},
        {"path": "../escape.md", "outcome": "solved"},
        {"path": "digest/procedure/fail-only.md", "outcome": "failed"},
        "not-a-dict",
    ]
    candidates = fast_track_candidates(records, muscle_prefix="digest/muscle/", workspace=tmp_path)
    assert candidates == ["digest/procedure/a.md"]


@pytest.mark.asyncio
async def test_usage_ingest_emits_candidates(tmp_path):
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    (tmp_path / "digest" / "procedure" / "a.md").write_text("body", encoding="utf-8")
    step = OpsUsageIngestStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=_store(tmp_path),
    )
    response = await step(RuntimeContext(records=[
        {"ts": "t", "path": "digest/procedure/a.md", "outcome": "solved"},
        {"ts": "t", "path": "digest/procedure/b.md", "outcome": "failed"},
    ]))

    assert response.success is True
    assert response.metadata["usage_ingest"]["fast_track_candidates"] == ["digest/procedure/a.md"]
