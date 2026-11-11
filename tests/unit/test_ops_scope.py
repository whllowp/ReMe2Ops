"""Scope factorization and the generalize operator."""

import json
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.generalize import OpsGeneralizeStep, parse_audit_reply
from reme.steps.ops.muscle_list import OpsMuscleListStep, scope_allows


def test_scope_allows_general_and_device():
    assert scope_allows("general", "d1") is True
    assert scope_allows("", "d1") is True  # default general
    assert scope_allows("device:d1", "d1") is True
    assert scope_allows("device:d2", "d1") is False


def _write_sop(tmp_path, rel, *, scope="general", body="runbook"):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frontmatter.dumps(frontmatter.Post(body, name="x", description="d", scope=scope)),
        encoding="utf-8",
    )
    return rel


@pytest.mark.asyncio
async def test_muscle_list_filters_by_device_scope(tmp_path):
    general = _write_sop(tmp_path, "digest/muscle/general-runbook.md", scope="general")
    own = _write_sop(tmp_path, "digest/muscle/host-a-only.md", scope="device:edge-a")
    other = _write_sop(tmp_path, "digest/muscle/host-b-only.md", scope="device:edge-b")

    step = OpsMuscleListStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)),
                             file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path))
    response = await step(RuntimeContext(for_device="edge-a"))

    paths = [f["path"] for f in response.metadata["files"]]
    assert general in paths and own in paths
    assert other not in paths
    assert response.metadata["for_device"] == "edge-a"

    # An unscoped pull (legacy sync) sees everything general + all devices.
    response_all = await step(RuntimeContext())
    assert len(response_all.metadata["files"]) == 3


AUDIT_JSON = json.dumps({"moves": [
    {"path": "digest/muscle/host-runbook.md", "action": "promote", "scope": "general", "reason": "generic"},
    {"path": "digest/muscle/general-2.md", "action": "demote", "scope": "device:edge-b", "device": "edge-b",
     "reason": "hostnames inside"},
    {"path": "digest/muscle/unknown-path.md", "action": "promote", "scope": "general", "reason": "invented"},
    {"path": "digest/muscle/general-2.md", "action": "keep", "scope": "general", "reason": "already right"},
]})


class FakeWrapper(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        return {"status": "completed", "error": None, "result": AUDIT_JSON}


def test_parse_audit_reply_drops_unknown_paths_and_bad_actions():
    moves = parse_audit_reply(AUDIT_JSON, {"digest/muscle/host-runbook.md", "digest/muscle/general-2.md"})
    assert [m["path"] for m in moves] == [
        "digest/muscle/host-runbook.md",
        "digest/muscle/general-2.md",
        "digest/muscle/general-2.md",
    ]
    assert moves[0]["scope"] == "general"
    assert moves[1]["scope"] == "device:edge-b"
    assert moves[2]["action"] == "keep"


@pytest.mark.asyncio
async def test_generalize_applies_scope_moves(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.generalize.llm_available", lambda step: True)
    host = _write_sop(tmp_path, "digest/muscle/host-runbook.md", scope="device:edge-a")
    general2 = _write_sop(tmp_path, "digest/muscle/general-2.md", scope="general")

    step = OpsGeneralizeStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)),
                             file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
                             agent_wrapper=FakeWrapper())
    async def fake_frontmatter_update(name, *, path, metadata):
        target = tmp_path / path
        post = frontmatter.loads(target.read_text(encoding="utf-8"))
        post.metadata.update(metadata)
        target.write_text(frontmatter.dumps(post), encoding="utf-8")
        return Mock(success=True, answer="stamped")

    step.run_job = AsyncMock(side_effect=fake_frontmatter_update)
    response = await step(RuntimeContext())

    assert response.metadata["generalize"]["moves"][0]["applied"] is True
    moved_general = frontmatter.loads((tmp_path / host).read_text(encoding="utf-8"))
    moved_device = frontmatter.loads((tmp_path / general2).read_text(encoding="utf-8"))
    assert moved_general.metadata["scope"] == "general"
    assert moved_device.metadata["scope"] == "device:edge-b"


@pytest.mark.asyncio
async def test_generalize_without_llm_degrades_to_report(tmp_path):
    _write_sop(tmp_path, "digest/muscle/a.md")
    step = OpsGeneralizeStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)),
                             file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
                             agent_wrapper=Mock(spec=BaseAgentWrapper))
    response = await step(RuntimeContext())
    assert response.metadata["generalize"]["skipped"] == "no llm configured"
    assert response.success is True
