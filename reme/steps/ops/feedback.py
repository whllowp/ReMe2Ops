"""Ops usage feedback: record whether a retrieved memory solved the problem."""

from ...components import R
from ..base_step import BaseStep
from .usage import (
    aggregate_usage,
    append_usage_record,
    normalize_outcome,
    now_iso,
    read_usage_records,
    usage_file_path,
)


@R.register("ops_feedback_step")
class OpsFeedbackStep(BaseStep):
    """Record a solved/failed outcome for one digest path (feeds cloud promotion)."""

    async def execute(self):
        assert self.context is not None
        path = str(self.context.get("path") or self.kwargs.get("path") or "").strip()
        outcome = normalize_outcome(self.context.get("outcome") or self.kwargs.get("outcome"))
        if not path:
            self.context.response.success = False
            self.context.response.answer = "Error: path is required"
            return self.context.response
        if outcome is None:
            self.context.response.success = False
            self.context.response.answer = "Error: outcome must be 'solved' or 'failed'"
            return self.context.response

        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        usage_path = usage_file_path(workspace, metadata_dir)
        record = {
            "ts": now_iso(),
            "path": path,
            "outcome": outcome,
            "query": str(self.context.get("query") or "").strip(),
            "session_id": str(self.context.get("session_id") or "").strip(),
        }
        append_usage_record(usage_path, record)
        stats = aggregate_usage(read_usage_records(usage_path)).get(path, {})

        self.context.response.answer = f"Recorded {outcome} for {path}"
        self.context.response.metadata["usage_file"] = usage_path.relative_to(workspace).as_posix()
        self.context.response.metadata["path_stats"] = {
            "solved": stats.get("solved", 0),
            "failed": stats.get("failed", 0),
            "success_rate": stats.get("success_rate", 0.0),
        }
        return self.context.response
