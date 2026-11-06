"""Ops muscle promotion: threshold scan and the promote step contract."""

from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.promote import OpsMusclePromoteStep, promotion_candidates


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
        **kwargs,
    )


@pytest.mark.asyncio
async def promote_step_moves_and_stamps_frontmatter(tmp_path):
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
async def promote_step_validations(tmp_path, path, expected_error):
    (tmp_path / "digest" / "procedure").mkdir(parents=True)
    step = _step(tmp_path)
    step.run_job = AsyncMock()
    response = await step(RuntimeContext(path=path))
    assert response.success is False
    assert expected_error in response.answer
    step.run_job.assert_not_awaited()


@pytest.mark.asyncio
async def promote_step_propagates_move_failure(tmp_path):
    source = tmp_path / "digest" / "procedure" / "a.md"
    source.parent.mkdir(parents=True)
    source.write_text("body", encoding="utf-8")
    step = _step(tmp_path)
    step.run_job = AsyncMock(return_value=Mock(success=False, answer="dst exists"))
    response = await step(RuntimeContext(path="digest/procedure/a.md"))
    assert response.success is False
    assert "dst exists" in response.answer
    assert step.run_job.await_count == 1
