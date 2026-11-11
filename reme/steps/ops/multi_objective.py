"""Multi-objective muscle promotion with Tchebycheff decomposition.

Harmony-Harness-style decomposition, scaled down to one decision: which
long-term SOPs deserve muscle status. Each candidate carries three normalized
objectives — success rate, usage volume, cross-device breadth — and a weighted
Tchebycheff scalarization keeps pressure on whichever objective sits furthest
below the running ideal point. A candidate is promoted when its worst weighted
shortfall is small; trading one objective away for a cheaper other is therefore
not accepted the way a weighted sum would accept it.

The ideal point persists across runs (``metadata/ops_promotion/ideal_point.json``)
so ``z*`` tracks the best SOP ever observed instead of being re-derived each
night.
"""

import json
from pathlib import Path

OBJECTIVES = ("rate", "solved", "breadth")

DEFAULT_WEIGHTS = {"rate": 0.4, "solved": 0.3, "breadth": 0.3}
# z* tracks the best RAW objective values ever observed (Harmony Harness:
# z*_j = max_H f_j(H)); floors of 1 keep the division defined before any
# evidence exists and make the utopia point (1, 1, 1) the starting reference.
IDEAL_FLOOR = {"rate": 1.0, "solved": 1.0, "breadth": 1.0}


def ideal_point_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the persisted ideal point."""
    return Path(workspace) / metadata_dir / "ops_promotion" / "ideal_point.json"


def load_ideal_point(path: Path) -> dict:
    """Load the running ideal point, falling back to floors."""
    ideal = dict(IDEAL_FLOOR)
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = {}
        for key in OBJECTIVES:
            value = stored.get(key)
            if isinstance(value, (int, float)) and value > 0:
                ideal[key] = float(value)
    return ideal


def save_ideal_point(path: Path, ideal: dict) -> None:
    """Persist the ideal point so later runs inherit the best ever seen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: float(ideal.get(key, IDEAL_FLOOR.get(key, 1.0))) for key in OBJECTIVES}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_ideal_point(ideal: dict, vectors: list[dict]) -> dict:
    """Grow z* toward the best objective values observed (never shrinks)."""
    updated = dict(ideal)
    for vector in vectors:
        for key in OBJECTIVES:
            value = float(vector.get(key) or 0.0)
            if value > updated.get(key, 0.0):
                updated[key] = value
    return updated


def objective_vector(stats: dict, *, solved_cap: float = 1.0, total_devices: int = 0) -> dict:
    """Extract the three RAW objectives for one usage-stat row.

    - ``rate``: solved share of all recorded outcomes, in [0, 1].
    - ``solved``: raw successful-use count.
    - ``breadth``: raw distinct-device count.

    ``solved_cap`` / ``total_devices`` are accepted for call compatibility but
    no longer clamp: the ideal point carries the scale, not a synthetic cap.
    """
    rate = min(max(float(stats.get("success_rate") or 0.0), 0.0), 1.0)
    solved = int(stats.get("solved") or 0)
    devices = int(stats.get("devices") or 0)
    return {"rate": rate, "solved": solved, "breadth": devices}


def tchebycheff_score(vector: dict, ideal: dict, weights: dict) -> float:
    """Weighted Tchebycheff: the largest weighted shortfall below z* (lower=better)."""
    worst = 0.0
    for key in OBJECTIVES:
        weight = float(weights.get(key, 0.0))
        if weight <= 0.0:
            continue
        reference = max(float(ideal.get(key, IDEAL_FLOOR.get(key, 1.0))), 1e-6)
        shortfall = weight * (1.0 - float(vector.get(key) or 0.0) / reference)
        if shortfall > worst:
            worst = shortfall
    return worst


def total_device_population(usage_stats: dict) -> int:
    """Distinct devices seen anywhere in the aggregate."""
    devices: set[str] = set()
    for stats in usage_stats.values():
        for device in stats.get("device_list") or []:
            devices.add(str(device))
    return len(devices)


def rank_promotion_candidates(
    usage_stats: dict,
    *,
    ideal: dict,
    weights: dict,
    min_solved: int,
    min_success_rate: float,
    score_threshold: float,
    procedure_dir: str,
) -> list[dict]:
    """Score procedure SOPs; return feasible candidates ordered best-first.

    Hard floors (``min_solved`` / ``min_success_rate``) stay as a feasibility
    prefilter — the evidence floor — while ordering and acceptance come from
    the Tchebycheff score against ``score_threshold``.
    """
    total_devices = total_device_population(usage_stats)
    floor_prefix = f"{procedure_dir.strip('/')}/"
    candidates: list[dict] = []
    for path, stats in usage_stats.items():
        if not path.startswith(floor_prefix):
            continue
        solved = int(stats.get("solved") or 0)
        rate = float(stats.get("success_rate") or 0.0)
        if solved < min_solved or rate < min_success_rate:
            continue
        vector = objective_vector(stats)
        score = tchebycheff_score(vector, ideal, weights)
        candidates.append({
            "path": path,
            "score": round(score, 4),
            "promotable": score <= score_threshold,
            **{f"{key}": round(vector[key], 4) for key in OBJECTIVES},
            "solved": solved,
            "success_rate": rate,
        })
    candidates.sort(key=lambda item: item["score"])
    return candidates
