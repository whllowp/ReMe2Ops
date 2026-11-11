"""Edge-side muscle sync: push usage deltas up, mirror cloud muscle SOPs down.

The mirror lives at ``digest/<digest_dir>/muscle/`` in the edge workspace; the
background index watcher picks up writes and deletions automatically, so a
synced SOP becomes instantly searchable on the reflex tier. Remote mtimes are
tracked in a local manifest (not the filesystem mtime, which changes on write).
"""

import json
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .cloud import CloudClient, cloud_settings
from .connector_client import default_device_id
from .usage import read_usage_records, usage_file_path

SYNC_STATE_FILE = "sync_state.json"


def load_manifest(path: Path) -> dict:
    """Load the muscle mirror manifest; tolerate a missing or corrupted file."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Persist the muscle mirror manifest atomically enough for a singleton edge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def plan_muscle_sync(
    remote_files: dict[str, dict],
    manifest: dict,
    local_paths: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Diff remote muscle files against the local mirror manifest.

    Returns ``(to_write, to_remove, unchanged)`` as workspace-relative paths.
    """
    to_write = []
    for path, meta in remote_files.items():
        known = manifest.get(path)
        if known is not None and known.get("mtime_ns") == meta.get("mtime_ns") and path in local_paths:
            continue
        to_write.append(path)
    to_remove = sorted(set(manifest) - set(remote_files))
    unchanged = sorted(set(remote_files) & set(manifest) - set(to_write))
    unchanged = [p for p in unchanged if p in local_paths]
    return to_write, to_remove, unchanged


@R.register("ops_muscle_sync_step")
class OpsMuscleSyncStep(BaseStep):
    """Two-way sync: usage deltas to the cloud, muscle SOPs from the cloud."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path.resolve()
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        host, port, timeout = cloud_settings(self)

        usage_path = usage_file_path(workspace, metadata_dir)
        sync_state_path = workspace / metadata_dir / "ops_feedback" / SYNC_STATE_FILE
        manifest_path = workspace / metadata_dir / "ops_muscle_manifest.json"

        pushed = 0
        written: list[str] = []
        removed: list[str] = []
        unchanged = 0
        errors: list[str] = []

        sync_state = load_manifest(sync_state_path)  # same tolerant JSON loader
        manifest = load_manifest(manifest_path)
        records = read_usage_records(usage_path)
        cursor = max(int(sync_state.get("synced_count") or 0), 0)
        delta = records[max(cursor, 0):]

        try:
            device_id = str(
                self.context.get("device_id")
                or self.kwargs.get("device_id")
                or default_device_id(),
            )
            async with CloudClient(host, port, timeout) as cloud:
                if delta:
                    # source_device makes the cloud breadth objective per-device.
                    resp = await cloud.post("ops_usage_ingest", {"records": delta, "source_device": device_id})
                    if resp.get("success", False):
                        pushed = len(delta)
                        sync_state["synced_count"] = len(records)
                        save_manifest(sync_state_path, sync_state)
                    else:
                        errors.append(f"usage ingest failed: {resp.get('answer') or 'unknown error'}")

                # Scope-aware pull: general muscle + this device's own scope only.
                resp = await cloud.post("muscle_list", {"for_device": device_id})
                if not resp.get("success", False):
                    errors.append(f"muscle_list failed: {resp.get('answer') or 'unknown error'}")
                else:
                    remote_list = (resp.get("metadata") or {}).get("files") or []
                    remote_files = {str(item.get("path") or ""): item for item in remote_list if item.get("path")}
                    muscle_root = workspace / muscle_rel
                    local_paths = (
                        {p.relative_to(workspace).as_posix() for p in muscle_root.rglob("*.md") if p.is_file()}
                        if muscle_root.is_dir()
                        else set()
                    )
                    to_write, to_remove, kept = plan_muscle_sync(remote_files, manifest, local_paths)
                    for rel in to_write:
                        target = workspace / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(remote_files[rel].get("content") or "", encoding="utf-8")
                        manifest[rel] = {"mtime_ns": remote_files[rel].get("mtime_ns"), "size": remote_files[rel].get("size")}
                        written.append(rel)
                    for rel in to_remove:
                        target = workspace / rel
                        try:
                            target.unlink()
                        except OSError as exc:
                            errors.append(f"remove failed for {rel}: {exc}")
                            continue
                        manifest.pop(rel, None)
                        removed.append(rel)
                    unchanged = len(kept)
                    if written or removed:
                        save_manifest(manifest_path, manifest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cloud unreachable at {host}:{port}: {type(exc).__name__}: {exc}")

        self.context.response.success = not errors
        self.context.response.answer = (
            f"Muscle sync pushed {pushed} usage record(s); mirror +{len(written)} / -{len(removed)} / "
            f"={unchanged} ({muscle_rel})"
        )
        self.context.response.metadata.update(
            {
                "pushed": pushed,
                "written": written,
                "removed": removed,
                "unchanged": unchanged,
                "pending_usage": max(len(records) - (sync_state.get("synced_count") or 0), 0)
                if errors
                else 0,
                "cloud": f"{host}:{port}",
                "errors": errors,
            },
        )
        return self.context.response
