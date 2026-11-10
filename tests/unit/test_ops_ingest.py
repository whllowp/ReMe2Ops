"""Ops cloud ingest seed: unit validation and dream-state seeding."""

from unittest.mock import Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import DreamState
from reme.steps.ops.ingest import OpsIngestSeedStep, normalize_ingest_units

UNIT_A = {
    "name": "disk-full-runbook",
    "bucket": "procedure",
    "summary": "How to clear a full disk without losing logs.",
    "paths": ["daily/2026-11-05/incident.md"],
}


def test_normalize_ingest_units_accepts_and_cleans():
    accepted, rejected = normalize_ingest_units(
        [
            UNIT_A,
            {"name": "u2", "bucket": "unknown-bucket", "summary": "s", "paths": ["daily/x.md"]},
            {"name": "", "bucket": "wiki", "summary": "s", "paths": ["daily/x.md"]},
            {"name": "u4", "bucket": "wiki", "summary": "", "paths": ["daily/x.md"]},
            {"name": "u5", "bucket": "wiki", "summary": "s", "paths": []},
            "not-a-dict",
        ],
    )
    assert [unit["name"] for unit in accepted] == ["disk-full-runbook", "u2"]
    assert accepted[1]["bucket"] == "wiki"  # unknown -> wiki
    assert len(rejected) == 4


def test_normalize_ingest_units_refuses_muscle_direct_writes():
    accepted, rejected = normalize_ingest_units(
        [{**UNIT_A, "name": "sneaky", "bucket": "muscle"}],
    )
    assert accepted == []
    assert len(rejected) == 1
    assert "promotion-only" in rejected[0]


def test_normalize_ingest_units_requires_list():
    accepted, rejected = normalize_ingest_units("nope")
    assert accepted == []
    assert rejected == ["units must be a list"]


@pytest.mark.asyncio
async def test_seed_step_writes_dream_state_into_context(tmp_path):
    step = OpsIngestSeedStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )
    context = RuntimeContext(units=[UNIT_A], source_device="edge-01", date="2026-11-06")
    response = await step(context)

    assert response.success is True
    assert response.metadata["accepted"] == 1
    assert response.metadata["source_device"] == "edge-01"
    state = DreamState.model_validate(context["dream"])
    assert [unit["name"] for unit in state.units] == ["disk-full-runbook"]
    assert state.changed_paths == ["daily/2026-11-05/incident.md"]
    assert state.processed_paths == state.changed_paths
    assert state.date == "2026-11-06"


@pytest.mark.asyncio
async def test_seed_step_rejects_empty_payload(tmp_path):
    step = OpsIngestSeedStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )
    context = RuntimeContext(units=[], source_device="edge-01")
    response = await step(context)

    assert response.success is False
    assert context["dream"]["units"] == []
