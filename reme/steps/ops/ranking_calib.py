"""Cloud-side ranking-weight dreaming calibration (MATM §3.2 LTRT analogue).

MATM's LTRT ranks retrievals by marginal utility — the success difference with
and without retrieval (``l = s_aug - s_base``) — and notes the labels can be
produced by the incremental build process itself. The ops analogue: the
offline replay already records which SOP actually solved each event, so the
retrieval-order labels come for free. This module replays the usage worlds
under a grid of candidate boost vectors and scores each by a deterministic
proxy objective:

    of the solved events that hit a would-be-muscle path, the fraction where
    that path also ranks top-1 among the would-be-muscle set — the ranking
    proxy being ``solved_count * (1 + usage_boost) - failed_count *
    conflict_penalty`` (cumulative outcome counts standing in for lexical
    scores — an acknowledged approximation; replay has no BM25/vector layer).

The boost vector therefore matters exactly where it should: a higher
``usage_boost`` reinforces the path that keeps solving (high-solved-first
worlds are monotonically non-worse under it), a higher ``conflict_penalty``
demotes paths that accumulated failures. Candidates are reduced to the
non-dominated front over ``(objective max, boost L1 norm min)`` — same
utility, less intervention preferred — and the front head is deployed to
``metadata/ops_policy/ranking.json``, which travels to edges through the
existing ``muscle_list`` -> ``muscle_sync`` channel (no new route).

Shared-Response contract: the step runs last in ``ops_policy_steps`` after
replay/evolve/calibrate peers that share one Response — only
``metadata["ranking_calib"]`` is written.
"""

import json
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .replay import build_session_trees
from .usage import now_iso, read_usage_records, usage_file_path

RANKING_POLICY_FILE = "ranking.json"
# Proxy promotion floors, deliberately looser than the live promotion policy:
# a path with failures must still join the ranked set, otherwise
# conflict_penalty would have nothing to demote and stay uncalibratable.
PROMOTION_MIN_SOLVED = 2
PROMOTION_MIN_RATE = 0.5

# Deterministic candidate boost vectors (dimensionless retrieval weights).
CALIBRATION_GRID: tuple[dict, ...] = (
    {"usage_boost": 0.0, "confirmed_boost": 0.0, "conflict_penalty": 0.0},
    {"usage_boost": 0.05, "confirmed_boost": 0.05, "conflict_penalty": 0.15},
    {"usage_boost": 0.1, "confirmed_boost": 0.05, "conflict_penalty": 0.15},
    {"usage_boost": 0.1, "confirmed_boost": 0.1, "conflict_penalty": 0.1},
    {"usage_boost": 0.2, "confirmed_boost": 0.05, "conflict_penalty": 0.2},
)


def ranking_policy_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the deployed ranking policy (cloud and edge snapshot)."""
    return Path(workspace) / metadata_dir / "ops_policy" / RANKING_POLICY_FILE


def boost_l1(boosts: dict) -> float:
    """L1 norm of a boost vector — the intervention-size axis of the front."""
    return round(sum(abs(float(boosts.get(key) or 0.0)) for key in ("usage_boost", "confirmed_boost", "conflict_penalty")), 4)


def replay_ranking_objective(trees: list[list[dict]], boosts: dict) -> dict:
    """Walk the worlds once; return the top-1 fraction of muscle solved events.

    Promotion floors are fixed (the calibrated vector must not move the muscle
    boundary — that is the policy genome's job), only the proxy ranking among
    would-be-muscle paths depends on the boosts. Deterministic ties break to
    the lexicographically first path.
    """
    usage_boost = float(boosts.get("usage_boost") or 0.0)
    conflict_penalty = float(boosts.get("conflict_penalty") or 0.0)
    min_solved = PROMOTION_MIN_SOLVED
    min_rate = PROMOTION_MIN_RATE

    counts: dict[str, dict[str, int]] = {}
    muscle: set[str] = set()
    solved_events = 0
    top1_events = 0

    def proxy_key(path: str) -> tuple[float, str]:
        """Ranking key: highest proxy score first, ties to the first path name."""
        entry = counts.get(path) or {}
        score = int(entry.get("solved") or 0) * (1.0 + usage_boost) - int(entry.get("failed") or 0) * conflict_penalty
        return (-score, path)

    for tree in trees:
        for record in tree:
            path = str(record.get("path") or "").strip()
            outcome = str(record.get("outcome") or "").strip().lower()
            if not path or outcome not in {"solved", "failed"}:
                continue
            entry = counts.setdefault(path, {"solved": 0, "failed": 0})
            entry[outcome] += 1
            if not (entry["solved"] >= min_solved and entry["solved"] / (entry["solved"] + entry["failed"]) >= min_rate):
                continue
            newly_promoted = path not in muscle
            muscle.add(path)
            if outcome != "solved" or newly_promoted:
                continue
            solved_events += 1
            if min(muscle, key=proxy_key) == path:
                top1_events += 1

    objective = round(top1_events / solved_events, 4) if solved_events else 0.0
    return {"objective": objective, "solved_events": solved_events, "top1_events": top1_events}


def pareto_front(candidates: list[dict]) -> list[dict]:
    """Non-dominated candidates over (objective max, boost L1 norm min)."""
    return sorted(
        [
            candidate
            for candidate in candidates
            if not any(
                other["objective"] >= candidate["objective"]
                and other["boost_l1"] <= candidate["boost_l1"]
                and (other["objective"] > candidate["objective"] or other["boost_l1"] < candidate["boost_l1"])
                for other in candidates
            )
        ],
        key=lambda candidate: (-candidate["objective"], candidate["boost_l1"]),
    )


def calibrate_ranking_params(trees: list[list[dict]], grid: list[dict] | tuple[dict, ...]) -> dict:
    """Score every candidate boost vector; return the non-dominated front.

    Returns ``{"frontier": [...], "winner": {...} | None, "evaluated": int,
    "solved_events": int}``; the winner is the front head (highest objective,
    smallest intervention). ``None`` when the grid is empty or the worlds
    produced no muscle solved events to learn from.
    """
    evaluated: list[dict] = []
    total_events = 0
    for boosts in grid or ():
        vector = {key: float(boosts.get(key) or 0.0) for key in ("usage_boost", "confirmed_boost", "conflict_penalty")}
        result = replay_ranking_objective(trees, vector)
        total_events = max(total_events, result["solved_events"])
        evaluated.append({"boosts": vector, "boost_l1": boost_l1(vector), **result})
    frontier = pareto_front(evaluated)
    return {
        "frontier": frontier,
        "winner": frontier[0] if frontier else None,
        "evaluated": len(evaluated),
        "solved_events": total_events,
    }


@R.register("ops_ranking_calibrate_step")
class OpsRankingCalibrateStep(BaseStep):
    """Dream-calibrate the retrieval boost weights and deploy them for edges.

    Replays the cloud usage aggregate under :data:`CALIBRATION_GRID`, keeps the
    non-dominated front, and writes ``metadata/ops_policy/ranking.json``
    (``{"deployed": {boosts}, "frontier": [...], "calibrated_at", "worlds"}``).
    Skips without writing when no usage worlds or no muscle solved events exist
    (nothing to learn from — never deploy a guess). The file travels to edges
    via ``muscle_list`` -> ``muscle_sync``; the search step picks it up with
    the same explicit-kwargs > deployed-snapshot > built-in-default contract
    as the promotion policy. Job success accounting stays with the cron chain:
    only ``metadata["ranking_calib"]`` is written.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path.resolve()
        metadata_dir = str(self.config_value("metadata_dir"))

        records = read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True))
        trees = build_session_trees(records)
        if not trees:
            self.context.response.metadata["ranking_calib"] = {
                "skipped": "no replay worlds: usage history has no session/query grouping",
                "path": None,
            }
            return self.context.response

        result = calibrate_ranking_params(trees, CALIBRATION_GRID)
        if result["winner"] is None or result["solved_events"] == 0:
            # No learnable evidence: deploying zero boosts would silently
            # disable the weights, so skip instead.
            self.context.response.metadata["ranking_calib"] = {
                "skipped": "no would-be-muscle solved events to calibrate on",
                "path": None,
            }
            return self.context.response

        payload = {
            "deployed": dict(result["winner"]["boosts"]),
            "objective": result["winner"]["objective"],
            "frontier": result["frontier"],
            "worlds": len(trees),
            "calibrated_at": now_iso(),
        }
        path = ranking_policy_path(workspace, metadata_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        self.context.response.metadata["ranking_calib"] = {
            "skipped": "",
            "path": path.relative_to(workspace).as_posix(),
            "deployed": payload["deployed"],
            "objective": payload["objective"],
            "frontier_size": len(result["frontier"]),
            "evaluated": result["evaluated"],
            "solved_events": result["solved_events"],
            "worlds": len(trees),
        }
        return self.context.response
