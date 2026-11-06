"""Cloud-side ingestion seed: turn an edge push into dream-integration state."""

from ...components import R
from ...enumeration import DreamBucketEnum
from ...schema import DreamState
from ..base_step import BaseStep
from ..evolve.dream.utils import daily_dir, state_from_context, store_state, today, workspace_dir

# Muscle memory is promotion-only; the cloud never accepts direct muscle writes.
_INGEST_BUCKETS = (DreamBucketEnum.PROCEDURE, DreamBucketEnum.PERSONAL, DreamBucketEnum.WIKI)


def normalize_ingest_units(raw_units, *, logger=None, step_name: str = "ops_ingest_seed_step") -> tuple[list[dict], list[str]]:
    """Validate pushed units; return (accepted, rejection reasons).

    Unknown buckets route to wiki (matching extract behaviour); ``muscle`` is
    rejected explicitly because it is reachable only through usage promotion.
    """
    accepted: list[dict] = []
    rejected: list[str] = []
    if not isinstance(raw_units, list):
        return [], ["units must be a list"]
    allowed = {bucket.value for bucket in _INGEST_BUCKETS}
    all_buckets = {bucket.value for bucket in DreamBucketEnum}
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            rejected.append(f"units[{index}] not an object")
            continue
        name = str(raw.get("name") or "").strip()
        summary = str(raw.get("summary") or "").strip()
        bucket = str(raw.get("bucket") or "").strip()
        paths = [str(p).strip() for p in raw.get("paths") or [] if str(p).strip()]
        if not name or not summary or not paths:
            rejected.append(f"units[{index}] ({name or 'unnamed'}) missing name/summary/paths")
            continue
        if bucket in all_buckets and bucket not in allowed:
            rejected.append(f"units[{index}] ({name}) bucket '{bucket}' is promotion-only; refused")
            if logger is not None:
                logger.warning(f"[{step_name}] unit {name!r} pushed bucket '{bucket}'; muscle is promotion-only")
            continue
        if bucket not in allowed:
            if logger is not None:
                logger.warning(f"[{step_name}] unit {name!r} unknown bucket '{bucket}'; routing to wiki")
            bucket = DreamBucketEnum.WIKI.value
        accepted.append({"name": name, "bucket": bucket, "summary": summary, "paths": paths})
    return accepted, rejected


@R.register("ops_ingest_seed_step")
class OpsIngestSeedStep(BaseStep):
    """Seed dream state from an edge push so dream_integrate_step runs unchanged.

    Edge source paths (``daily/...``) do not exist in the cloud workspace; they
    are preserved as provenance so digest ``## Sources`` wikilinks still point
    back at the originating short-term evidence.
    """

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)  # preserve any caller-provided dream fields
        raw_units = self.context.get("units")
        source_device = str(self.context.get("source_device") or "").strip()

        accepted, rejected = normalize_ingest_units(raw_units, logger=self.logger, step_name=self.name)
        state.units = accepted
        state.workspace = str(workspace_dir(self))
        state.daily_dir = daily_dir(self)
        state.date = today(self, str(self.context.get("date") or "")) if self.context.get("date") else state.date or today(self)
        state.changed_paths = sorted({path for unit in accepted for path in unit.get("paths", [])})
        state.processed_paths = list(state.changed_paths)
        state.failed_paths = []
        store_state(self, state)

        if not accepted:
            self.context.response.success = False
            self.context.response.answer = "No valid units to ingest"
            self.context.response.metadata["rejected"] = rejected
            self.context.response.metadata["source_device"] = source_device
            return self.context.response

        self.context.response.answer = (
            f"Seeded {len(accepted)} unit(s) from device '{source_device or 'unknown'}' "
            f"for integration ({len(rejected)} rejected)"
        )
        self.context.response.metadata["accepted"] = len(accepted)
        self.context.response.metadata["rejected"] = rejected
        self.context.response.metadata["source_device"] = source_device
        return self.context.response
