"""Cloud-side muscle promotion: usage-qualified SOPs become muscle memory."""

import json
from pathlib import Path

from ...components import R
from ...components.as_decision_model.base_as_decision_model import BaseAsDecisionModel
from ...enumeration import ComponentEnum
from ..base_step import BaseStep, Ref
from .connector import broadcast_muscle_update
from .judge import judge_sops
from .multi_objective import (
    DEFAULT_WEIGHTS,
    ideal_point_path,
    load_ideal_point,
    rank_promotion_candidates,
    save_ideal_point,
    update_ideal_point,
)
from .usage import aggregate_usage, now_iso, read_usage_records, usage_file_path


def policy_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the deployed promotion policy (Dream-RSI style)."""
    return Path(workspace) / metadata_dir / "ops_policy" / "policy.json"


def load_deployed_policy(workspace, metadata_dir: str) -> dict:
    """Read the policy written by ops_policy_evolve; empty when absent/corrupt.

    The replay-evolved policy is the *runtime default*; explicit job kwargs
    (or config YAML) always win over it.
    """
    path = policy_path(workspace, metadata_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    params = data.get("params") if isinstance(data, dict) else None
    return params if isinstance(params, dict) else {}


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
        promotion_score = self.context.get("promotion_score")
        if promotion_score is None:
            promotion_score = self.kwargs.get("promotion_score")
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
        stamp = {"memory_type": "muscle", "promoted_at": now_iso(), "promoted_from": path,
                 "scope": "general"}
        if judge_probability is not None:
            stamp["promoted_by"] = "decision-model"
            stamp["judge_probability"] = round(float(judge_probability), 4)
        elif promotion_score is not None:
            stamp["promoted_by"] = "tchebycheff"
            stamp["promotion_score"] = round(float(promotion_score), 4)
        await self.run_job("frontmatter_update", path=dst, metadata=stamp)

        # The reflex-card step (if enabled) consumes this context entry.
        self.context["promoted"] = {"from": path, "to": dst}
        await broadcast_muscle_update(self, [dst])
        self.context.response.answer = f"Promoted {path} → {dst}"
        self.context.response.metadata["promoted"] = {"from": path, "to": dst}
        return self.context.response


@R.register("ops_promotion_scan_step")
class OpsPromotionScanStep(BaseStep):
    """Promote procedure SOPs by multi-objective Tchebycheff score.

    Objectives (rate / solved / breadth) are normalized and scalarized against
    a persisted ideal point; a candidate is promoted when its worst weighted
    shortfall is at most ``score_threshold``. Runtime defaults may come from
    the replay-evolved ``policy.json``; explicit kwargs win over it. The old
    min_solved/min_success_rate pair remains as a feasibility prefilter, and
    ``decision_fast_track`` still lets the decision model promote confident
    below-floor candidates.
    """

    as_decision_model = Ref(BaseAsDecisionModel, ComponentEnum.AS_DECISION_MODEL, optional=True)

    def _param(self, key: str, default):
        """Explicit kwargs win over the deployed policy; policy over builtin."""
        explicit = self.kwargs.get(key)
        if explicit is not None:
            return explicit
        return self._policy.get(key, default)

    async def execute(self):
        assert self.context is not None
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        procedure_dir = str(self.kwargs.get("procedure_dir") or f"{digest_dir}/procedure").strip("/")
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        self._policy = load_deployed_policy(workspace, metadata_dir)

        min_solved = max(int(self._param("min_solved", 5)), 1)
        min_success_rate = float(self._param("min_success_rate", 0.8))
        score_threshold = float(self._param("score_threshold", 0.45))
        weights = {
            "rate": float(self._param("lambda_rate", DEFAULT_WEIGHTS["rate"])),
            "solved": float(self._param("lambda_solved", DEFAULT_WEIGHTS["solved"])),
            "breadth": float(self._param("lambda_breadth", DEFAULT_WEIGHTS["breadth"])),
        }
        max_promotions = max(int(self.kwargs.get("max_promotions_per_run", 3)), 1)
        decision_fast_track = bool(self.kwargs.get("decision_fast_track", False))
        fast_track_min_solved = max(int(self.kwargs.get("fast_track_min_solved", 1)), 1)
        max_fast_tracks = max(int(self.kwargs.get("max_fast_tracks_per_run", 2)), 1)

        usage_stats = aggregate_usage(read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True)))
        ideal_path = ideal_point_path(workspace, metadata_dir)
        ideal = load_ideal_point(ideal_path)

        candidates = rank_promotion_candidates(
            usage_stats,
            ideal=ideal,
            procedure_dir=procedure_dir,
            min_solved=min_solved,
            min_success_rate=min_success_rate,
            score_threshold=score_threshold,
            weights=weights,
        )
        vectors = [{key: candidate[key] for key in ("rate", "solved", "breadth")} for candidate in candidates]
        ideal = update_ideal_point(ideal, vectors)
        save_ideal_point(ideal_path, ideal)

        promotable = [candidate for candidate in candidates if candidate["promotable"]][:max_promotions]

        promoted: list[dict] = []
        skipped: list[dict] = []
        promoted_paths: set[str] = set()
        for candidate in promotable:
            path = candidate["path"]
            if not (workspace / path).is_file():
                skipped.append({"path": path, "reason": "file missing"})
                continue
            response = await self.run_job("ops_muscle_promote", path=path, promotion_score=candidate["score"])
            entry = {"path": path, "score": candidate["score"], "solved": candidate["solved"],
                     "success_rate": candidate["success_rate"], "breadth": candidate["breadth"]}
            if response.success:
                promoted.append(entry)
                promoted_paths.add(path)
            else:
                skipped.append({**entry, "reason": str(response.answer)})

        fast_tracked: list[dict] = []
        if decision_fast_track:
            prefix = f"{procedure_dir.strip('/')}/"
            scored_paths = {candidate["path"] for candidate in candidates}
            below = [
                (path, stats)
                for path, stats in usage_stats.items()
                if path.startswith(prefix)
                and path not in scored_paths
                and int(stats.get("solved") or 0) >= fast_track_min_solved
                and (workspace / path).is_file()
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
            f"Promotion scan: {len(candidates)} feasible, {len(promotable)} above Tchebycheff bar, "
            f"{len(promoted)} promoted, {len(skipped)} skipped "
            f"(min_solved={min_solved}, min_success_rate={min_success_rate}, score_threshold={score_threshold})"
        )
        if self._policy:
            answer += f"; policy from replay (score={self._policy.get('replay_score')})"
        if decision_fast_track:
            answer += f"; fast-track judged with decision model, {len(fast_tracked)} promoted"
        self.context.response.answer = answer
        self.context.response.metadata["promotion"] = {
            "candidates": candidates,
            "promoted": promoted,
            "fast_tracked": fast_tracked,
            "skipped": skipped,
            "ideal_point": {key: round(value, 4) for key, value in ideal.items()},
            "policy_applied": bool(self._policy),
        }
        return self.context.response
