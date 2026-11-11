"""Replay simulator over usage history; promotion policies evaluated by dreaming.

Dream-RSI (arXiv 2609.14858 §3), scaled to the ops memory system: the cloud
usage aggregate is the accumulated discovery history. Each session of
query -> retrieved path -> solved/failed outcome is one small world. A
*promotion policy* (thresholds, weights, score bar) can be evaluated offline by
replaying sessions chronologically and asking: had this policy been live, which
SOPs would have become muscle memory, and how many later solves would the
reflex tier have served?

The replay objective mirrors Dream-RSI Eq. 1: a normalized hypervolume over the
(quality, cost) Pareto points obtained by sweeping a budget preference, minus a
penalty for false promotions. The best policy is deployed to
``metadata/ops_policy/policy.json`` and picked up by the promotion scan as its
runtime default — closing a policy-level RSI loop where online usage funds
cheap offline policy evaluation.
"""

import json
from pathlib import Path

from .multi_objective import DEFAULT_WEIGHTS

# Budget sweep: how lenient the score bar is during replay (Dream-RSI's rho).
REPLAY_BARS = (0.25, 0.4, 0.55, 0.7, 0.85)
POLICY_VERSIONS = 4  # M: current + 3 candidates per offline phase


def policy_path(workspace, metadata_dir: str) -> Path:
    """Where the deployed policy lives."""
    return Path(workspace) / metadata_dir / "ops_policy" / "policy.json"


def build_session_trees(records: list[dict]) -> list[list[dict]]:
    """Group records into per-session worlds, each a chronology of events.

    A session is one incident thread: it hits candidate SOPs and either solves
    or fails with them. Sessions without an id fall back to their query text,
    and events are ordered by timestamp. Records without any grouping key are
    dropped — they cannot form a world.
    """
    groups: dict[str, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("session_id") or "").strip() or str(record.get("query") or "").strip()
        if not key:
            continue
        groups.setdefault(key, []).append(record)
    trees: list[list[dict]] = []
    for events in groups.values():
        events.sort(key=lambda r: str(r.get("ts") or ""))
        trees.append(events)
    return trees


def policy_params(policy: dict) -> dict:
    """Normalize one policy candidate into replay parameters with defaults."""
    weights_in = policy.get("weights") if isinstance(policy.get("weights"), dict) else {}
    return {
        "name": str(policy.get("name") or "candidate"),
        "min_solved": max(int(policy.get("min_solved", 1)), 1),
        "min_success_rate": float(policy.get("min_success_rate", 0.5)),
        "solved_cap": max(float(policy.get("solved_cap", 10)), 1.0),
        "score_bar": float(policy.get("score_bar", 0.55)),
        "weights": {
            "rate": float(weights_in.get("rate", DEFAULT_WEIGHTS["rate"])),
            "solved": float(weights_in.get("solved", DEFAULT_WEIGHTS["solved"])),
            "breadth": float(weights_in.get("breadth", DEFAULT_WEIGHTS["breadth"])),
        },
    }


def replay_once(trees: list[list[dict]], params: dict) -> dict:
    """Replay every world under one policy; return its (quality, cost) counts.

    The replayed universe is a single global promotion state: walking worlds in
    order, a path enters the would-be-muscle set when its running evidence
    passes the policy floors and Tchebycheff bar (using fixed replay caps so
    the simulation is deterministic). Afterwards, a solved event on a
    would-be-muscle path is a reflex hit; a failed one is a false reflex.
    """
    from .multi_objective import (  # local import to avoid cycles in tests
        IDEAL_FLOOR,
        objective_vector,
        tchebycheff_score,
    )

    min_solved = params["min_solved"]
    min_rate = params["min_success_rate"]
    bar = params["score_bar"]
    cap = params["solved_cap"]
    weights = params["weights"]

    stats: dict[str, dict] = {}
    muscle: set[str] = set()
    promotions = 0
    reflex_hits = 0
    false_reflex = 0
    total_solved = 0
    total_failed = 0

    for tree in trees:
        for record in tree:
            path = str(record.get("path") or "").strip()
            outcome = str(record.get("outcome") or "").strip().lower()
            if not path or outcome not in {"solved", "failed"}:
                continue
            entry = stats.setdefault(path, {"solved": 0, "failed": 0, "success_rate": 0.0, "devices": 0})
            entry[outcome] += 1
            total = entry["solved"] + entry["failed"]
            entry["success_rate"] = round(entry["solved"] / total, 4)

            if outcome == "solved":
                total_solved += 1
                if path in muscle:
                    reflex_hits += 1
            else:
                total_failed += 1
                if path in muscle:
                    false_reflex += 1

            # Evidence-driven promotion check after each event (like the live
            # scan, which runs after each sync): promote once, never demote.
            if path not in muscle and entry["solved"] >= min_solved and entry["success_rate"] >= min_rate:
                vector = objective_vector(entry, solved_cap=cap, total_devices=IDEAL_FLOOR["breadth"] * 10)
                if tchebycheff_score(vector, IDEAL_FLOOR, weights) <= bar:
                    muscle.add(path)
                    promotions += 1

    return {
        "promotions": promotions,
        "reflex_hits": reflex_hits,
        "false_reflex": false_reflex,
        "total_solved": total_solved,
        "total_failed": total_failed,
        "muscle_size": len(muscle),
    }


def replay_score(trees: list[list[dict]], params: dict, *, parallelism_penalty: float = 0.1) -> dict:
    """Dream-RSI objective: hypervolume over the budget sweep, minus penalties.

    Sweeping the score bar across ``REPLAY_BARS`` yields (cost, quality) points
    where cost is promotions triggered and quality is reflex hits earned. The
    normalized hypervolume of the non-dominated frontier rewards policies that
    reach high quality with few promotions; the penalty term punishes the share
    of reflex misses among would-be reflex outcomes (the analogue of the paper's
    parallelism penalty: pay per bad decision, not per attempt).
    """
    counts_by_bar: list[tuple[float, dict]] = []
    for bar in REPLAY_BARS:
        swept = dict(params)
        swept["score_bar"] = bar
        counts = replay_once(trees, swept)
        counts_by_bar.append((bar, counts))

    # Non-dominated quality-cost frontier: quality = reflex hits, cost = promotions.
    points = []
    for bar, counts in counts_by_bar:
        points.append({"bar": bar, "cost": counts["promotions"], "quality": counts["reflex_hits"]})
    frontier: list[dict] = []
    for point in points:
        dominated = any(
            other["cost"] <= point["cost"] and other["quality"] >= point["quality"]
            and (other["cost"] < point["cost"] or other["quality"] > point["quality"])
            for other in points
        )
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda p: p["cost"])

    max_cost = max((point["cost"] for point in points), default=0)
    max_quality = max((point["quality"] for point in points), default=0)
    hypervolume = 0.0
    prev_cost = 0
    prev_quality = 0.0
    if max_quality > 0:
        for point in frontier + [{"cost": max_cost, "quality": max_quality}]:
            hypervolume += (point["quality"] - prev_quality) * (max_cost - point["cost"])
            prev_cost = point["cost"]
            prev_quality = point["quality"]
        hypervolume /= max(max_quality * max_cost, 1)

    mid = counts_by_bar[len(counts_by_bar) // 2][1]
    reflex_outcomes = mid["reflex_hits"] + mid["false_reflex"]
    false_rate = (mid["false_reflex"] / reflex_outcomes) if reflex_outcomes else 0.0
    score = round(hypervolume - parallelism_penalty * false_rate, 4)

    return {
        "score": score,
        "hypervolume": round(hypervolume, 4),
        "false_rate": round(false_rate, 4),
        "frontier": frontier,
        "counts": {str(bar): counts for bar, counts in counts_by_bar},
    }


def load_deployed_policy_file(workspace, metadata_dir: str) -> dict:
    """Full policy record (params + replay score + provenance), or {}."""
    path = policy_path(workspace, metadata_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_deployed_policy(workspace, metadata_dir: str, record: dict) -> Path:
    """Persist the winning policy record atomically enough for a singleton."""
    path = policy_path(workspace, metadata_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
