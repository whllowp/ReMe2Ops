"""Ops push step: delivery success defers nothing; failures defer for retry."""

from unittest.mock import Mock

import pytest

from reme.components import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.schema import DreamState
from reme.steps.ops.push import OpsPushStep, unit_paths

UNITS = [
    {"name": "a", "bucket": "procedure", "summary": "sa", "paths": ["daily/2026-11-05/a.md"]},
    {"name": "b", "bucket": "wiki", "summary": "sb", "paths": ["daily/2026-11-05/b.md"]},
]


def test_unit_paths_union():
    assert unit_paths(UNITS) == ["daily/2026-11-05/a.md", "daily/2026-11-05/b.md"]
    assert unit_paths([]) == []


def _step(tmp_path):
    return OpsPushStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), source_device="edge-01")


def _context(units):
    state = DreamState(date="2026-11-06", units=units, changed_paths=unit_paths(units), processed_paths=[])
    return RuntimeContext(dream=state.model_dump())


@pytest.mark.asyncio
async def push_success_keeps_paths_checkpointable(tmp_path, monkeypatch):
    cloud_payloads = []

    class StubCloud:
        def __init__(self, host, port, timeout=8.0, transport=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            cloud_payloads.append((action, payload))
            return {
                "success": True,
                "answer": "integrated",
                "metadata": {
                    "dream": {
                        "integrate_results": [{"action": "CREATE", "target_path": "digest/procedure/a.md"}],
                        "nodes_created": ["digest/procedure/a.md"],
                        "nodes_updated": [],
                        "failed_units": [],
                    },
                },
            }

    monkeypatch.setattr("reme.steps.ops.push.CloudClient", StubCloud)
    step = _step(tmp_path)
    context = _context(UNITS)
    response = await step(context)

    assert response.success is True
    action, payload = cloud_payloads[0]
    assert action == "cloud_ingest"
    assert payload["source_device"] == "edge-01"
    assert [unit["name"] for unit in payload["units"]] == ["a", "b"]
    state = DreamState.model_validate(context["dream"])
    assert state.failed_paths == []
    assert state.nodes_created == ["digest/procedure/a.md"]
    assert response.metadata["push"]["delivered"] == 2


@pytest.mark.asyncio
async def push_transport_failure_defers_all_paths(tmp_path, monkeypatch):
    class DownCloud:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("cloud down")

    monkeypatch.setattr("reme.steps.ops.push.CloudClient", DownCloud)
    step = _step(tmp_path)
    context = _context(UNITS)
    response = await step(context)

    assert response.success is False
    state = DreamState.model_validate(context["dream"])
    assert state.failed_paths == ["daily/2026-11-05/a.md", "daily/2026-11-05/b.md"]
    assert response.metadata["push"]["delivered"] == 0


@pytest.mark.asyncio
async def push_cloud_rejection_defers_all_paths(tmp_path, monkeypatch):
    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            return {"success": False, "answer": "no llm on cloud"}

    monkeypatch.setattr("reme.steps.ops.push.CloudClient", StubCloud)
    step = _step(tmp_path)
    context = _context(UNITS)
    response = await step(context)

    assert response.success is False
    assert "no llm on cloud" in response.answer
    state = DreamState.model_validate(context["dream"])
    assert len(state.failed_paths) == 2


@pytest.mark.asyncio
async def push_partial_cloud_integration_defers_only_failed(tmp_path, monkeypatch):
    class StubCloud:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            return {
                "success": True,
                "answer": "integrated",
                "metadata": {
                    "dream": {
                        "integrate_results": [{"action": "CREATE", "target_path": "digest/procedure/a.md"}],
                        "nodes_created": [],
                        "nodes_updated": [],
                        "failed_units": [{"name": "b", "bucket": "wiki", "summary": "sb", "paths": ["daily/2026-11-05/b.md"]}],
                    },
                },
            }

    monkeypatch.setattr("reme.steps.ops.push.CloudClient", StubCloud)
    step = _step(tmp_path)
    context = _context(UNITS)
    response = await step(context)

    assert response.success is True
    state = DreamState.model_validate(context["dream"])
    assert state.failed_paths == ["daily/2026-11-05/b.md"]
    assert response.metadata["push"]["delivered"] == 1


@pytest.mark.asyncio
async def push_without_units_is_a_noop(tmp_path, monkeypatch):
    class Unused:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cloud must not be reached")

    monkeypatch.setattr("reme.steps.ops.push.CloudClient", Unused)
    step = _step(tmp_path)
    context = _context([])
    response = await step(context)

    assert response.success is True
    assert response.metadata["push"] == {"units": 0, "delivered": 0, "failed_units": 0}
