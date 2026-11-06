"""Cloud-side muscle promotion: usage-qualified SOPs become muscle memory."""

from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .usage import aggregate_usage, now_iso, read_usage_records, usage_file_path


def promotion_candidates(
    usage_stats: dict[str, dict],
    *,
    procedure_dir: str,
    min_solved: int,
    min_success_rate: float,
) -> list[tuple[str, dict]]:
    """Return procedure SOPs whose usage meets the muscle thresholds, best first."""
    prefix = f"{procedure_dir.strip('/')}/"
    candidates: list[tuple[str, dict]] = []
    for path, stats in usage_stats.items():
        if not path.startswith(prefix):
            continue
        if int(stats.get("solved") or 0) < min_solved:
            continue
        if float(stats.get("success_rate") or 0.0) < min_success_rate:
            continue
        candidates.append((path, stats))
    candidates.sort(key=lambda item: (-int(item[1].get("solved") or 0), -float(item[1].get("success_rate") or 0.0)))
    return candidates


@R.register("ops_muscle_promote_step")
class OpsMusclePromoteStep(BaseStep):
    """Promote one digest SOP into the muscle bucket (move + frontmatter stamp)."""

    async def execute(self):
        assert self.context is not None
        path = str(self.context.get("path") or self.kwargs.get("path") or "").strip()
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_dir = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        workspace = self.file_store.workspace_path

        if not path:
            self.context.response.success = False
            self.context.response.answer = "Error: path is required"
            return self.context.response
        if not path.startswith(f"{digest_dir}/") or path.startswith(f"{muscle_dir}/"):
            self.context.response.success = False
            self.context.response.answer = (
                f"Error: only long-term digest SOPs can be promoted (got {path!r}); muscle is reachable only via usage"
            )
            return self.context.response
        if not (workspace / path).is_file():
            self.context.response.success = False
            self.context.response.answer = f"Error: {path} does not exist"
            return self.context.response

        dst = f"{muscle_dir}/{Path(path).name}"
        move_response = await self.run_job("move", src_path=path, dst_path=dst, overwrite=True)
        if not move_response.success:
            self.context.response.success = False
            self.context.response.answer = f"Error: move failed: {move_response.answer}"
            return self.context.response
        await self.run_job(
            "frontmatter_update",
            path=dst,
            metadata={"memory_type": "muscle", "promoted_at": now_iso(), "promoted_from": path},
        )

        self.context.response.answer = f"Promoted {path} → {dst}"
        self.context.response.metadata["promoted"] = {"from": path, "to": dst}
        return self.context.response


@R.register("ops_promotion_scan_step")
class OpsPromotionScanStep(BaseStep):
    """Promote procedure SOPs whose aggregated usage meets the muscle thresholds."""

    async def execute(self):
        assert self.context is not None
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        procedure_dir = str(self.kwargs.get("procedure_dir") or f"{digest_dir}/procedure").strip("/")
        min_solved = max(int(self.kwargs.get("min_solved", 5)), 1)
        min_success_rate = float(self.kwargs.get("min_success_rate", 0.8))
        max_promotions = max(int(self.kwargs.get("max_promotions_per_run", 3)), 1)
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))

        usage_stats = aggregate_usage(read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True)))
        qualified = promotion_candidates(
            usage_stats,
            procedure_dir=procedure_dir,
            min_solved=min_solved,
            min_success_rate=min_success_rate,
        )

        promoted: list[dict] = []
        skipped: list[dict] = []
        for path, stats in qualified[:max_promotions]:
            if not (workspace / path).is_file():
                skipped.append({"path": path, "reason": "file missing"})
                continue
            response = await self.run_job("ops_muscle_promote", path=path)
            entry = {"path": path, "solved": stats.get("solved"), "success_rate": stats.get("success_rate")}
            if response.success:
                promoted.append(entry)
            else:
                skipped.append({**entry, "reason": str(response.answer)})

        self.context.response.answer = (
            f"Promotion scan: {len(qualified)} qualified, {len(promoted)} promoted, {len(skipped)} skipped "
            f"(min_solved={min_solved}, min_success_rate={min_success_rate})"
        )
        self.context.response.metadata["promotion"] = {
            "qualified": [{"path": p, **{k: v for k, v in s.items() if k != "last_used"}} for p, s in qualified],
            "promoted": promoted,
            "skipped": skipped,
        }
        return self.context.response
