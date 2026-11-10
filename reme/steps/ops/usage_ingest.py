"""Cloud-side usage ingestion: append edge usage deltas to the aggregate."""

from ...components import R
from ..base_step import BaseStep
from .usage import append_usage_record, normalize_outcome, usage_file_path


def fast_track_candidates(records: list[dict], *, muscle_prefix: str, workspace) -> list[str]:
    """Distinct solved paths from one ingest batch eligible for immediate judgment.

    Muscle-bucket paths are already reflexes and missing files are skipped;
    path traversal outside the workspace is dropped.
    """
    solved: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        path = str(record.get("path") or "").strip()
        if path and normalize_outcome(record.get("outcome")) == "solved" and path not in solved:
            solved.append(path)
    candidates = []
    for path in solved:
        if path.startswith(muscle_prefix):
            continue
        try:
            inside = (workspace / path).resolve().is_relative_to(workspace.resolve())
        except (OSError, ValueError):
            inside = False
        if inside and (workspace / path).is_file():
            candidates.append(path)
    return candidates


@R.register("ops_usage_ingest_step")
class OpsUsageIngestStep(BaseStep):
    """Append edge usage records to the cloud aggregate that drives promotion.

    Solved digest paths from the batch are emitted as
    ``context["fast_track_candidates"]`` for the decision-model judge step
    that follows in the ``ops_usage_ingest`` job.
    """

    async def execute(self):
        assert self.context is not None
        raw_records = self.context.get("records")
        if not isinstance(raw_records, list):
            self.context.response.success = False
            self.context.response.answer = "Error: records must be a list"
            return self.context.response

        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        usage_path = usage_file_path(workspace, metadata_dir, cloud=True)

        accepted, invalid = 0, 0
        for record in raw_records:
            if not isinstance(record, dict):
                invalid += 1
                continue
            path = str(record.get("path") or "").strip()
            outcome = normalize_outcome(record.get("outcome"))
            if not path or outcome is None:
                invalid += 1
                continue
            append_usage_record(
                usage_path,
                {
                    "ts": str(record.get("ts") or ""),
                    "path": path,
                    "outcome": outcome,
                    "query": str(record.get("query") or ""),
                    "session_id": str(record.get("session_id") or ""),
                    "source_device": str(self.context.get("source_device") or record.get("source_device") or ""),
                },
            )
            accepted += 1

        candidates = fast_track_candidates(
            raw_records,
            muscle_prefix=f"{digest_dir}/muscle/",
            workspace=workspace,
        )
        self.context["fast_track_candidates"] = candidates

        self.context.response.success = accepted > 0 or not raw_records
        self.context.response.answer = f"Appended {accepted} usage record(s) ({invalid} invalid)"
        self.context.response.metadata["usage_ingest"] = {
            "accepted": accepted,
            "invalid": invalid,
            "fast_track_candidates": candidates,
            "usage_file": usage_path.relative_to(workspace).as_posix(),
        }
        return self.context.response
