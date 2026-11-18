"""Dream-RSI policy loop for promotion: replay history, deploy the best policy.

``ops_replay`` evaluates candidate promotion policies offline against the
accumulated usage history (the replay simulator) and reports a ranking.
``ops_policy_evolve`` wraps it in the recursive loop: seed the candidate pool
with the deployed policy plus a deterministic jitter grid (a pluggable proposer
can replace the grid later), replay all versions, and deploy the winner to
``metadata/ops_policy/policy.json`` — the promotion scan reads it as runtime
defaults, so the improvement is live on the next scan. The deployed policy is
always among the candidates, hence never replaced by a worse one.
"""

import json

from ...components import R
from ..base_step import BaseStep
from .replay import (
    POLICY_VERSIONS,
    build_session_trees,
    load_deployed_policy_file,
    policy_params,
    replay_score,
    save_deployed_policy,
    policy_path,
)
from .usage import now_iso, read_usage_records, usage_file_path


def default_candidate_grid(current: dict | None, *, mutate_weights: bool = False) -> list[dict]:
    """Deterministic first-generation pool around the deployed policy.

    With ``mutate_weights`` on, the method space (Sage-Taiyi cognitive
    repertoire: reusable units are methods/heuristics, not episodes) joins the
    grid — λ weight mixes and ``solved_cap`` candidates lead the pool so the
    version cap cannot drop them.
    """
    base = policy_params(current or {"name": "seed"})
    candidates = [dict(base, name="deployed")]
    grid: list[tuple[str, list]] = []
    if mutate_weights:
        grid.extend(
            [
                ("weights", [
                    {"rate": 0.5, "solved": 0.3, "breadth": 0.2},
                    {"rate": 0.3, "solved": 0.4, "breadth": 0.3},
                    {"rate": 0.6, "solved": 0.3, "breadth": 0.1},
                ]),
                ("solved_cap", [5.0, 20.0]),
            ],
        )
    grid.extend(
        [
            ("min_solved", [1, 2, 3, 5]),
            ("score_bar", [0.45, 0.55, 0.7]),
            ("min_success_rate", [0.5, 0.8]),
        ],
    )
    seen = {json_key(base)}
    for key, values in grid:
        for value in values:
            candidate = dict(base)
            candidate[key] = value
            candidate["name"] = f"grid-{key}={json.dumps(value, sort_keys=True) if isinstance(value, dict) else value}"
            marker = json_key(candidate)
            if marker not in seen:
                seen.add(marker)
                candidates.append(candidate)
    return candidates[: max(POLICY_VERSIONS * 3, 12)]


def json_key(params: dict) -> str:
    """Stable identity for a param set, used to dedupe the pool."""
    import json

    return json.dumps(policy_params(params), sort_keys=True, ensure_ascii=False)


WEIGHT_JITTER = 0.2  # ±20% multiplicative jitter before renormalization
CAP_JITTER = 1.0
_WEIGHT_FLOOR = 0.01


def normalize_weights(raw: dict) -> dict:
    """λ mix summing to exactly 1 (Tchebycheff decomposition semantics).

    Each weight is floored, renormalized, rounded to 4 decimals, and the last
    weight is the exact complement — so ``sum == 1`` holds to float precision
    and every weight stays positive.
    """
    values = {key: max(float(raw.get(key) or 0.0), 1e-6) for key in ("rate", "solved", "breadth")}
    total = sum(values.values())
    floored = {key: max(value / total, _WEIGHT_FLOOR) for key, value in values.items()}
    rescale = 1.0 / sum(floored.values())
    rate = round(floored["rate"] * rescale, 4)
    solved = round(floored["solved"] * rescale, 4)
    breadth = round(1.0 - rate - solved, 4)
    if breadth < _WEIGHT_FLOOR:  # keep every lambda positive; steal from the larger one
        delta = _WEIGHT_FLOOR - breadth
        breadth = _WEIGHT_FLOOR
        if rate >= solved:
            rate = round(rate - delta, 4)
        else:
            solved = round(solved - delta, 4)
    return {"rate": rate, "solved": solved, "breadth": breadth}


def mutate_params(params: dict, rng, *, mutate_weights: bool = False) -> dict:
    """One dreaming candidate: deterministic jitter around the incumbent.

    With ``mutate_weights`` on, the λ mix gets a multiplicative jitter followed
    by :func:`normalize_weights` (the sum stays exactly 1, so the Tchebycheff
    decomposition semantics are unchanged) and ``solved_cap`` gets a bounded
    jitter clamped to ``[1, 50]``. The rng draw sequence is fixed, so the same
    seed always yields the same candidate.
    """
    import copy

    candidate = copy.deepcopy(params)
    candidate["name"] = f"mut-{rng.randrange(10 ** 6):06d}"
    if rng.random() < 0.6:
        candidate["min_solved"] = max(int(round(float(candidate.get("min_solved", 1)))) + rng.choice((-1, 1)), 1)
    if rng.random() < 0.6:
        candidate["score_bar"] = round(min(max(float(candidate.get("score_bar", 0.55)) + rng.uniform(-0.05, 0.05), 0.1), 0.9), 4)
    if rng.random() < 0.4:
        candidate["min_success_rate"] = round(min(max(float(candidate.get("min_success_rate", 0.5)) + rng.uniform(-0.1, 0.1), 0.0), 1.0), 4)
    if mutate_weights:
        if rng.random() < 0.5:
            weights = candidate.get("weights") if isinstance(candidate.get("weights"), dict) else {}
            candidate["weights"] = normalize_weights({
                key: float(weights.get(key) or 0.0) * (1.0 + rng.uniform(-WEIGHT_JITTER, WEIGHT_JITTER))
                for key in ("rate", "solved", "breadth")
            })
        if rng.random() < 0.4:
            candidate["solved_cap"] = round(
                min(max(float(candidate.get("solved_cap", 10)) + rng.uniform(-CAP_JITTER, CAP_JITTER), 1.0), 50.0),
                4,
            )
    return candidate


@R.register("ops_replay_step")
class OpsReplayStep(BaseStep):
    """Score candidate promotion policies against recorded usage worlds.

    With ``rounds > 1`` this becomes the Dream-RSI dreaming phase proper:
    each round mutates candidates around the incumbent (round-seeded RNG, so
    runs are reproducible), replays, and tracks the round winner. Scores never
    regress — the deployed policy stays in every pool — and a per-round curve
    lands in ``metadata/ops_policy/evolution_log.jsonl``.
    """

    async def execute(self):
        assert self.context is not None
        import datetime as _dt
        import random

        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        usage_path = usage_file_path(workspace, metadata_dir, cloud=True)
        records = read_usage_records(usage_path)
        trees = build_session_trees(records)
        if not trees:
            self.context.response.success = False
            self.context.response.answer = "No replay worlds: usage history has no session/query grouping"
            return self.context.response

        rounds = max(int(self.context.get("rounds") or self.kwargs.get("rounds") or 1), 1)
        raw = self.context.get("policies") or self.kwargs.get("policies") or []
        explicit = isinstance(raw, list) and bool(raw)
        # Sage-Taiyi cognitive repertoire: mutate the method space (λ weights +
        # solved_cap), not only the three thresholds. Deterministic, no LLM.
        mutate_weights = bool(self.kwargs.get("mutate_weights", False))
        # Evidence basis follows the live promotion scan (ops_promotion_cron
        # sets the same value): the dreaming evaluation must tune thresholds on
        # the counting basis the scan actually applies, or the deployed
        # policy targets a semantics it will never see again.
        independent_only = bool(self.kwargs.get("independent_only", False))

        log_path = workspace / metadata_dir / "ops_policy" / "evolution_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        deployed = load_deployed_policy_file(workspace, metadata_dir)
        deployed_score = float(deployed.get("replay_score") or float("-inf"))
        incumbent = deployed.get("params") or {"name": "seed"}
        round_curve: list[dict] = []
        overall_best: dict | None = None

        with log_path.open("a", encoding="utf-8") as log:
            for round_index in range(1, rounds + 1):
                rng = random.Random(round_index * 7919)
                if explicit and round_index == 1:
                    pool = [policy_params(item) for item in raw]
                else:
                    base = (overall_best or {}).get("params") or incumbent
                    pool = [policy_params(dict(base, name="incumbent"))]
                    pool.extend(mutate_params(base, rng, mutate_weights=mutate_weights) for _ in range(max(POLICY_VERSIONS - 1, 1)))

                ranking = []
                for params in pool:
                    normalized = policy_params(params)
                    result = replay_score(trees, normalized, independent_only=independent_only)
                    ranking.append({"name": normalized["name"], "params": normalized, **result})
                ranking.sort(key=lambda item: item["score"], reverse=True)
                winner = ranking[0]
                if winner["score"] > deployed_score:
                    deployed_score = winner["score"]
                    overall_best = winner
                    redeployed = True
                else:
                    redeployed = False
                entry = {
                    "round": round_index,
                    "candidates": len(pool),
                    "best_candidate": winner["name"],
                    "best_score": winner["score"],
                    "deployed_score": round(deployed_score, 4),
                    "redeployed": redeployed,
                    "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                }
                round_curve.append(entry)
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if overall_best is None:
            overall_best = {"name": str(incumbent.get("name")), "params": incumbent, "score": deployed_score}
        self.context["replay_winner"] = overall_best
        self.context["replay_redeploy"] = bool(
            overall_best.get("params")
            and overall_best["score"] > float(load_deployed_policy_file(workspace, metadata_dir).get("replay_score") or float("-inf")),
        )

        self.context.response.answer = (
            f"Dreamed {rounds} round(s) over {len(trees)} world(s); "
            f"best={overall_best['name']} score={overall_best['score']}; "
            f"curve at {log_path.name}"
        )
        self.context.response.metadata["replay"] = {
            "worlds": len(trees),
            "rounds": round_curve,
            "winner": overall_best,
            "redeploy": self.context["replay_redeploy"],
        }
        return self.context.response


@R.register("ops_policy_evolve_step")
class OpsPolicyEvolveStep(BaseStep):
    """Deploy the replay winner as the promotion policy when it beats the incumbent."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        winner = self.context.get("replay_winner")
        if winner is None:
            self.context.response.success = False
            self.context.response.answer = "Error: no replay winner in context (run ops_replay_step first)"
            return self.context.response

        previous = load_deployed_policy_file(workspace, metadata_dir)
        if not self.context.get("replay_redeploy"):
            # The deployed policy is among the candidates, so a non-better
            # winner means the incumbent still holds (Dream-RSI: pi never
            # regresses); keep its file and version untouched.
            self.context.response.answer = (
                f"Kept deployed policy '{(previous.get('params') or {}).get('name')}' "
                f"(replay score {previous.get('replay_score')}; best candidate {winner['score']})"
            )
            self.context.response.metadata["policy"] = {"kept": True, **previous}
            return self.context.response
        record = {
            "params": winner["params"],
            "replay_score": winner["score"],
            "hypervolume": winner.get("hypervolume"),
            "false_rate": winner.get("false_rate"),
            "deployed_at": now_iso(),
            "policy_version": int(previous.get("policy_version") or 0) + 1,
        }
        path = save_deployed_policy(workspace, metadata_dir, record)
        self.context.response.answer = (
            f"Deployed policy '{winner['params']['name']}' v{record['policy_version']} "
            f"(replay score {winner['score']})"
        )
        self.context.response.metadata["policy"] = {
            "path": path.relative_to(workspace).as_posix(),
            **record,
        }
        return self.context.response
