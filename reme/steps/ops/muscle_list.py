"""Cloud-side muscle listing: what an edge mirror should hold."""

from ...components import R
from ..base_step import BaseStep


@R.register("ops_muscle_list_step")
class OpsMuscleListStep(BaseStep):
    """List muscle-bucket SOP files (path, mtime, content) for edge mirrors."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        root = workspace / muscle_rel

        max_bytes = self.max_file_bytes()
        files: list[dict] = []
        skipped: list[str] = []
        if root.is_dir():
            for file_path in sorted(root.rglob("*.md")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(workspace).as_posix()
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

        self.context.response.answer = f"{len(files)} muscle file(s) available ({len(skipped)} skipped oversize)"
        self.context.response.metadata["muscle_dir"] = muscle_rel
        self.context.response.metadata["files"] = files
        self.context.response.metadata["skipped"] = skipped
        return self.context.response
