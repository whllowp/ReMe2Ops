"""Cloud source reverse index: reverse citation view, shared sources, idempotence."""

from pathlib import Path
from unittest.mock import Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.source_index import (
    OpsSourceIndexStep,
    affected_nodes,
    build_source_index,
    load_source_index,
    shared_source_nodes,
    source_index_path,
)

PROCEDURE_A = "digest/procedure/disk-cleanup.md"
PROCEDURE_B = "digest/procedure/reboot.md"
WIKI_C = "digest/wiki/gateway.md"

SHARED_SOURCE = "daily/2026-11-17/incident.md"
UNIQUE_SOURCE = "daily/2026-11-16/followup.md"
DANGLING_SOURCE = "daily/2026-11-10/virtual.md"

BODY_A = f"""# Disk cleanup

Order matters.

## Sources

From [[{SHARED_SOURCE}]] and [[{UNIQUE_SOURCE}]]; see [[{SHARED_SOURCE}]] again.
"""

BODY_B = f"""# Reboot

## Sources

Also [[{SHARED_SOURCE}]], plus the dangling [[{DANGLING_SOURCE}]].
"""

BODY_C = "# Gateway\n\nNo provenance section.\n"


def _write_node(tmp_path: Path, rel: str, body: str, description: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(body, name=Path(rel).stem, description=description)), encoding="utf-8")


def _step(tmp_path: Path):
    return OpsSourceIndexStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


# ----- pure functions ---------------------------------------------------------


def test_build_source_index_reverse_view_with_dangling_virtual_targets():
    index = build_source_index({PROCEDURE_A: BODY_A, PROCEDURE_B: BODY_B, WIKI_C: BODY_C})
    # two nodes cite the same daily source; dedup within a node
    assert index[SHARED_SOURCE] == [PROCEDURE_A, PROCEDURE_B]
    assert index[UNIQUE_SOURCE] == [PROCEDURE_A]
    # dangling daily/ targets are indexed as-is: virtual nodes are intentional
    assert index[DANGLING_SOURCE] == [PROCEDURE_B]
    assert WIKI_C not in {node for nodes in index.values() for node in nodes}  # no-Sources node cites nothing


def test_build_source_index_keys_and_values_sorted():
    index = build_source_index(
        {
            PROCEDURE_B: f"## Sources\n\n[[{SHARED_SOURCE}]] [[{UNIQUE_SOURCE}]]",
            PROCEDURE_A: f"## Sources\n\n[[{UNIQUE_SOURCE}]]",
        },
    )
    assert list(index) == sorted(index)  # source keys sorted
    assert index[UNIQUE_SOURCE] == [PROCEDURE_A, PROCEDURE_B]  # citing nodes sorted


def test_shared_source_nodes_marks_shared_premises():
    index = build_source_index({PROCEDURE_A: BODY_A, PROCEDURE_B: BODY_B})
    shared = shared_source_nodes(index)
    assert shared == {SHARED_SOURCE: [PROCEDURE_A, PROCEDURE_B]}
    assert UNIQUE_SOURCE not in shared and DANGLING_SOURCE not in shared
    assert shared_source_nodes(index, min_nodes=3) == {}
    assert shared_source_nodes({}, min_nodes=2) == {}


def test_affected_nodes_returns_targeted_revision_set():
    index = build_source_index({PROCEDURE_A: BODY_A, PROCEDURE_B: BODY_B})
    assert affected_nodes(index, SHARED_SOURCE) == [PROCEDURE_A, PROCEDURE_B]  # both revisited on source change
    assert affected_nodes(index, UNIQUE_SOURCE) == [PROCEDURE_A]
    assert affected_nodes(index, DANGLING_SOURCE) == [PROCEDURE_B]
    assert affected_nodes(index, "daily/2026-01-01/unknown.md") == []
    assert affected_nodes({}, SHARED_SOURCE) == []
    assert affected_nodes(index, f" {SHARED_SOURCE} ") == [PROCEDURE_A, PROCEDURE_B]  # trimmed lookup


def test_load_source_index_tolerates_corruption(tmp_path):
    assert load_source_index(tmp_path / "nope.json") == {}
    corrupt = tmp_path / "broken.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_source_index(corrupt) == {}


# ----- step -------------------------------------------------------------------


def _seed_workspace(tmp_path: Path) -> None:
    _write_node(tmp_path, PROCEDURE_A, BODY_A, "Disk cleanup SOP")
    _write_node(tmp_path, PROCEDURE_B, BODY_B, "Reboot SOP")
    _write_node(tmp_path, WIKI_C, BODY_C, "Gateway notes")


@pytest.mark.asyncio
async def test_step_writes_reverse_index_and_shared_report(tmp_path):
    _seed_workspace(tmp_path)
    response = await _step(tmp_path)(RuntimeContext())

    index = load_source_index(source_index_path(tmp_path, "metadata"))
    assert index == {
        SHARED_SOURCE: [PROCEDURE_A, PROCEDURE_B],
        UNIQUE_SOURCE: [PROCEDURE_A],
        DANGLING_SOURCE: [PROCEDURE_B],
    }
    meta = response.metadata["source_index"]
    assert meta["sources"] == 3 and meta["links"] == 4
    assert meta["index"] == "metadata/ops_source_index.json"
    assert meta["shared"] == {SHARED_SOURCE: [PROCEDURE_A, PROCEDURE_B]}
    assert meta["shared_count"] == 1
    # shared-Response contract: answer/success stay with the cron chain
    assert response.answer == ""
    assert response.success is True


@pytest.mark.asyncio
async def test_step_rerun_is_idempotent_same_input_same_json(tmp_path):
    _seed_workspace(tmp_path)
    step = _step(tmp_path)
    await step(RuntimeContext())
    first = source_index_path(tmp_path, "metadata").read_bytes()

    response2 = await step(RuntimeContext())
    second = source_index_path(tmp_path, "metadata").read_bytes()

    assert first == second  # byte-identical rebuild
    assert response2.metadata["source_index"]["sources"] == 3


@pytest.mark.asyncio
async def test_step_tolerates_corrupt_existing_index(tmp_path):
    _seed_workspace(tmp_path)
    index_path = source_index_path(tmp_path, "metadata")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{not valid json!!", encoding="utf-8")

    assert load_source_index(index_path) == {}
    response = await _step(tmp_path)(RuntimeContext())  # rebuild overwrites it

    assert response.metadata["source_index"]["sources"] == 3
    assert load_source_index(index_path)[SHARED_SOURCE] == [PROCEDURE_A, PROCEDURE_B]


@pytest.mark.asyncio
async def test_step_without_digest_or_sources_writes_empty_index(tmp_path):
    (tmp_path / "digest").mkdir()
    response = await _step(tmp_path)(RuntimeContext())

    meta = response.metadata["source_index"]
    assert meta["sources"] == 0 and meta["links"] == 0
    assert meta["shared"] == {} and meta["shared_count"] == 0
    assert load_source_index(source_index_path(tmp_path, "metadata")) == {}


# ----- config wiring ----------------------------------------------------------


def test_cloud_wiring_job_and_maintenance_slot():
    cloud = _load_config("ops_cloud.yaml")
    job = cloud["jobs"]["ops_source_index"]
    assert job["backend"] == "base"
    assert job["enable_serve"] is True
    assert [step["backend"] for step in job["steps"]] == ["ops_source_index_step"]
    maintenance = [step["backend"] for step in cloud["jobs"]["ops_maintenance_cron"]["steps"]]
    assert maintenance[-1] == "ops_source_index_step"  # sees the post-repair graph
    assert maintenance.index("ops_source_index_step") > maintenance.index("ops_evidence_step")
    assert maintenance.index("ops_source_index_step") > maintenance.index("ops_link_repair_step")
