"""Ops muscle sync: mirror diff planning, manifest handling, and the sync step."""

import json
from unittest.mock import Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.muscle_sync import (
    OpsMuscleSyncStep,
    load_manifest,
    plan_muscle_sync,
    save_manifest,
)


def _remote(path, mtime_ns, content="body"):
    return {"path": path, "mtime_ns": mtime_ns, "size": len(content), "content": content}


def test_plan_muscle_sync_new_updated_unchanged_removed():
    remote = {
        "digest/muscle/new.md": _remote("digest/muscle/new.md", 100),
        "digest/muscle/updated.md": _remote("digest/muscle/updated.md", 200),
        "digest/muscle/same.md": _remote("digest/muscle/same.md", 300),
    }
    manifest = {
        "digest/muscle/updated.md": {"mtime_ns": 111},
        "digest/muscle/same.md": {"mtime_ns": 300},
        "digest/muscle/gone.md": {"mtime_ns": 999},
    }
    to_write, to_remove, unchanged = plan_muscle_sync(remote, manifest, set(remote))
    assert to_write == ["digest/muscle/new.md", "digest/muscle/updated.md"]
    assert to_remove == ["digest/muscle/gone.md"]
    assert unchanged == ["digest/muscle/same.md"]


def test_plan_muscle_sync_skips_known_files_missing_locally():
    remote = {"digest/muscle/a.md": _remote("digest/muscle/a.md", 100)}
    manifest = {"digest/muscle/a.md": {"mtime_ns": 100}}
    to_write, to_remove, unchanged = plan_muscle_sync(remote, manifest, local_paths=set())
    # Manifest agrees with remote but the file vanished from disk: re-download.
    assert to_write == ["digest/muscle/a.md"]
    assert to_remove == []
    assert unchanged == []


def test_manifest_roundtrip_and_corruption(tmp_path):
    path = tmp_path / "manifest.json"
    save_manifest(path, {"digest/muscle/a.md": {"mtime_ns": 1}})
    assert load_manifest(path) == {"digest/muscle/a.md": {"mtime_ns": 1}}
    path.write_text("{broken", encoding="utf-8")
    assert load_manifest(path) == {}
    assert load_manifest(tmp_path / "missing.json") == {}


def _step(tmp_path):
    return OpsMuscleSyncStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
    )


async def _run_sync(tmp_path, monkeypatch, *, usage_response, list_response, existing_usage=()):
    """Run OpsMuscleSyncStep against a stubbed cloud."""
    calls = []

    class StubCloud:
        def __init__(self, host, port, timeout=15.0, transport=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, action, payload):
            calls.append((action, payload))
            return usage_response if action == "ops_usage_ingest" else list_response

    monkeypatch.setattr("reme.steps.ops.muscle_sync.CloudClient", StubCloud)
    usage_file = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    for record in existing_usage:
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        with usage_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    step = _step(tmp_path)
    response = await step(RuntimeContext())
    return response, calls, usage_file


@pytest.mark.asyncio
async def test_sync_pushes_usage_and_mirrors_muscle_files(tmp_path, monkeypatch):
    remote_a = _remote("digest/muscle/a.md", 100, content="---\nname: a\n---\nrunbook a")
    response, calls, _ = await _run_sync(
        tmp_path,
        monkeypatch,
        usage_response={"success": True, "answer": "ok"},
        list_response={"success": True, "metadata": {"files": [remote_a]}},
        existing_usage=[{"ts": "2026-11-06T08:00:00", "path": "digest/procedure/a.md", "outcome": "solved"}],
    )

    assert response.success is True
    pushed_call = next(payload for action, payload in calls if action == "ops_usage_ingest")
    assert len(pushed_call["records"]) == 1
    mirrored = tmp_path / "digest" / "muscle" / "a.md"
    assert mirrored.read_text(encoding="utf-8").endswith("runbook a")
    manifest = load_manifest(tmp_path / "metadata" / "ops_muscle_manifest.json")
    assert manifest["digest/muscle/a.md"]["mtime_ns"] == 100
    assert response.metadata["pushed"] == 1
    assert response.metadata["written"] == ["digest/muscle/a.md"]
    # Second sync without new data is a no-op.
    response2, calls2, _ = await _run_sync(
        tmp_path,
        monkeypatch,
        usage_response={"success": True, "answer": "ok"},
        list_response={"success": True, "metadata": {"files": [remote_a]}},
    )
    assert response2.metadata["pushed"] == 0
    assert response2.metadata["written"] == []
    assert response2.metadata["unchanged"] == 1
    assert all(action != "ops_usage_ingest" for action, _ in calls2)


@pytest.mark.asyncio
async def test_sync_removes_unlisted_mirrors_and_survives_cloud_failure(tmp_path, monkeypatch):
    mirror = tmp_path / "digest" / "muscle" / "stale.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("stale", encoding="utf-8")
    manifest_path = tmp_path / "metadata" / "ops_muscle_manifest.json"
    save_manifest(manifest_path, {"digest/muscle/stale.md": {"mtime_ns": 1}})

    class DownCloud:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("cloud down")

    monkeypatch.setattr("reme.steps.ops.muscle_sync.CloudClient", DownCloud)
    step = _step(tmp_path)
    response = await step(RuntimeContext())

    assert response.success is False
    assert any("cloud down" in error for error in response.metadata["errors"])
    assert mirror.exists()  # nothing is removed when the cloud cannot be reached

    response2, _, _ = await _run_sync(
        tmp_path,
        monkeypatch,
        usage_response={"success": True, "answer": "ok"},
        list_response={"success": True, "metadata": {"files": []}},
    )
    assert not mirror.exists()
    assert response2.metadata["removed"] == ["digest/muscle/stale.md"]


@pytest.mark.asyncio
async def test_sync_defers_usage_push_when_cloud_rejects(tmp_path, monkeypatch):
    response, calls, usage_file = await _run_sync(
        tmp_path,
        monkeypatch,
        usage_response={"success": False, "answer": "storage full"},
        list_response={"success": True, "metadata": {"files": []}},
        existing_usage=[{"ts": "2026-11-06T08:00:00", "path": "p.md", "outcome": "solved"}],
    )
    assert response.success is False
    assert any("storage full" in error for error in response.metadata["errors"])
    assert usage_file.read_text(encoding="utf-8").count("\n") == 1
    assert load_manifest(tmp_path / "metadata" / "ops_feedback" / "sync_state.json").get("synced_count") is None


def test_cloud_settings_defaults_and_context_override(tmp_path):
    from reme.steps.ops.cloud import cloud_settings

    step = OpsMuscleSyncStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        cloud_host="10.0.0.8",
        cloud_port=3444,
        cloud_timeout=15.0,
    )
    host, port, timeout = cloud_settings(step)
    assert (host, port, timeout) == ("10.0.0.8", 3444, 15.0)

    # A job-context request value wins over the YAML kwarg.
    step.context = RuntimeContext(cloud_port=4444)
    host, port, timeout = cloud_settings(step)
    assert (host, port, timeout) == ("10.0.0.8", 4444, 15.0)

    # Full defaults when neither is present.
    step.context = RuntimeContext()
    bare = OpsMuscleSyncStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    host, port, timeout = cloud_settings(bare)
    assert (host, port, timeout) == ("127.0.0.1", 2444, 8.0)
