"""Cloud-side muscle promotion: usage-qualified SOPs become muscle memory."""

from pathlib import Path

from ...components import R
from ...components.as_decision_model.base_as_decision_model import BaseAsDecisionModel
from ...enumeration import ComponentEnum
from ..base_step import BaseStep, Ref
from .judge import judge_sops
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
        judge_probability = self.context.get("judge_probability")
        if judge_probability is None:
            judge_probability = self.kwargs.get("judge_probability")
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
        stamp = {"memory_type": "muscle", "promoted_at": now_iso(), "promoted_from": path}
        if judge_probability is not None:
            stamp["promoted_by"] = "decision-model"
            stamp["judge_probability"] = round(float(judge_probability), 4)
        await self.run_job("frontmatter_update", path=dst, metadata=stamp)

        # The reflex-card step (if enabled) consumes this context entry.
        self.context["promoted"] = {"from": path, "to": dst}
        self.context.response.answer = f"Promoted {path} → {dst}"
        self.context.response.metadata["promoted"] = {"from": path, "to": dst}
        return self.context.response


@R.register("ops_promotion_scan_step")
class OpsPromotionScanStep(BaseStep):
    """Promote procedure SOPs whose aggregated usage meets the muscle thresholds.

    With ``decision_fast_track`` enabled, SOPs below the usage thresholds but
    with at least ``fast_track_min_solved`` successes get one decision-model
    verdict; a confident ``muscle_worthy`` promotes them immediately.
    """

    as_decision_model = Ref(BaseAsDecisionModel, ComponentEnum.AS_DECISION_MODEL, optional=True)

    async def execute(self):
        assert self.context is not None
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        procedure_dir = str(self.kwargs.get("procedure_dir") or f"{digest_dir}/procedure").strip("/")
        min_solved = max(int(self.kwargs.get("min_solved", 5)), 1)
        min_success_rate = float(self.kwargs.get("min_success_rate", 0.8))
        max_promotions = max(int(self.kwargs.get("max_promotions_per_run", 3)), 1)
        decision_fast_track = bool(self.kwargs.get("decision_fast_track", False))
        fast_track_min_solved = max(int(self.kwargs.get("fast_track_min_solved", 1)), 1)
        max_fast_tracks = max(int(self.kwargs.get("max_fast_tracks_per_run", 2)), 1)
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
        promoted_paths: set[str] = set()
        for path, stats in qualified[:max_promotions]:
            if not (workspace / path).is_file():
                skipped.append({"path": path, "reason": "file missing"})
                continue
            response = await self.run_job("ops_muscle_promote", path=path)
            entry = {"path": path, "solved": stats.get("solved"), "success_rate": stats.get("success_rate")}
            if response.success:
                promoted.append(entry)
                promoted_paths.add(path)
            else:
                skipped.append({**entry, "reason": str(response.answer)})

        fast_tracked: list[dict] = []
        if decision_fast_track:
            prefix = f"{procedure_dir.strip('/')}/"
            below = [
                (path, stats)
                for path, stats in usage_stats.items()
                if path.startswith(prefix)
                and path not in promoted_paths
                and int(stats.get("solved") or 0) >= fast_track_min_solved
                and (workspace / path).is_file()
                and (path, stats) not in qualified
            ]
            below.sort(key=lambda item: (-int(item[1].get("solved") or 0), -float(item[1].get("success_rate") or 0.0)))
            verdicts = await judge_sops(
                self,
                [path for path, _ in below[:max_fast_tracks]],
                min_solved=fast_track_min_solved,
            )
            for path, verdict in verdicts.items():
                if not verdict.get("promotable"):
                    continue
                response = await self.run_job(
                    "ops_muscle_promote",
                    path=path,
                    judge_probability=verdict.get("muscle_worthy"),
                )
                entry = {"path": path, "muscle_worthy": verdict.get("muscle_worthy")}
                if response.success:
                    fast_tracked.append(entry)
                else:
                    verdict["promote_error"] = str(response.answer)

        answer = (
            f"Promotion scan: {len(qualified)} qualified, {len(promoted)} promoted, {len(skipped)} skipped "
            f"(min_solved={min_solved}, min_success_rate={min_success_rate})"
        )
        if decision_fast_track:
            answer += f"; fast-track judged with decision model, {len(fast_tracked)} promoted"
        self.context.response.answer = answer
        self.context.response.metadata["promotion"] = {
            "qualified": [{"path": p, **{k: v for k, v in s.items() if k != "last_used"}} for p, s in qualified],
            "promoted": promoted,
            "fast_tracked": fast_tracked,
            "skipped": skipped,
        }
        return self.context.response
