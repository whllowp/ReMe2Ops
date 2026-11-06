"""Cloud-side usage ingestion: append edge usage deltas to the aggregate."""

from ...components import R
from ..base_step import BaseStep
from .usage import append_usage_record, normalize_outcome, usage_file_path


@R.register("ops_usage_ingest_step")
class OpsUsageIngestStep(BaseStep):
    """Append edge usage records to the cloud aggregate that drives promotion."""

    async def execute(self):
        assert self.context is not None
        raw_records = self.context.get("records")
        if not isinstance(raw_records, list):
            self.context.response.success = False
            self.context.response.answer = "Error: records must be a list"
            return self.context.response

        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
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

        self.context.response.success = accepted > 0 or not raw_records
        self.context.response.answer = f"Appended {accepted} usage record(s) ({invalid} invalid)"
        self.context.response.metadata["usage_ingest"] = {
            "accepted": accepted,
            "invalid": invalid,
            "usage_file": usage_path.relative_to(workspace).as_posix(),
        }
        return self.context.response
