"""Cloud-side muscle listing: what an edge mirror should hold.

Scope factorization (Harmony Harness §4.2): every muscle SOP carries
``scope: general`` or ``scope: device:<id>`` in its frontmatter. General
mechanisms download to every client; device-scoped ones only reach the named
device. A pull with ``for_device`` therefore assembles the per-device view
general ∪ own — the same composition idea the harness paper applies at
inference time, applied at sync time.
"""

import frontmatter

from ...components import R
from ..base_step import BaseStep

GENERAL_SCOPE = "general"


def scope_allows(scope: str, device_id: str) -> bool:
    """True when a SOP scope covers the requesting device."""
    scope = (scope or GENERAL_SCOPE).strip()
    if scope == GENERAL_SCOPE or not scope:
        return True
    if scope.startswith("device:"):
        return not device_id or scope.split(":", 1)[1].strip() == device_id
    return False


def file_scope(file_path, workspace) -> str:
    """Read the scope frontmatter of one muscle file (default: general)."""
    try:
        post = frontmatter.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - unreadable frontmatter degrades to general
        return GENERAL_SCOPE
    return str(post.metadata.get("scope") or GENERAL_SCOPE).strip()


@R.register("ops_muscle_list_step")
class OpsMuscleListStep(BaseStep):
    """List muscle SOPs visible to one device (scope-filtered)."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        for_device = str(self.context.get("for_device") or self.kwargs.get("for_device") or "").strip()
        root = workspace / muscle_rel

        max_bytes = self.max_file_bytes()
        files: list[dict] = []
        skipped: list[str] = []
        filtered = 0
        if root.is_dir():
            for file_path in sorted(root.rglob("*.md")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(workspace).as_posix()
                if not scope_allows(file_scope(file_path, workspace), for_device):
                    filtered += 1
                    continue
                stat = file_path.stat()
                if stat.st_size > max_bytes:
                    skipped.append(rel)
                    continue
                files.append(
                    {
                        "path": rel,
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "content": file_path.read_text(encoding="utf-8", errors="replace"),
                    },
                )

        scope_note = f" for device '{for_device}'" if for_device else ""
        self.context.response.answer = (
            f"{len(files)} muscle file(s) visible{scope_note} ({filtered} scope-filtered, {len(skipped)} skipped oversize)"
        )
        self.context.response.metadata["muscle_dir"] = muscle_rel
        self.context.response.metadata["for_device"] = for_device
        self.context.response.metadata["files"] = files
        self.context.response.metadata["skipped"] = skipped
        return self.context.response
