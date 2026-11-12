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


def default_candidate_grid(current: dict | None) -> list[dict]:
    """Deterministic first-generation pool around the deployed policy."""
    base = policy_params(current or {"name": "seed"})
    candidates = [dict(base, name="deployed")]
    grid = [
        ("min_solved", [1, 2, 3, 5]),
        ("score_bar", [0.45, 0.55, 0.7]),
        ("min_success_rate", [0.5, 0.8]),
    ]
    seen = {json_key(base)}
    for key, values in grid:
        for value in values:
            candidate = dict(base)
            candidate[key] = value
            candidate["name"] = f"grid-{key}={value}"
            marker = json_key(candidate)
            if marker not in seen:
                seen.add(marker)
                candidates.append(candidate)
    return candidates[: max(POLICY_VERSIONS * 3, 12)]


def json_key(params: dict) -> str:
    """Stable identity for a param set, used to dedupe the pool."""
    import json

    return json.dumps(policy_params(params), sort_keys=True, ensure_ascii=False)


def mutate_params(params: dict, rng) -> dict:
    """One dreaming candidate: deterministic jitter around the incumbent."""
    import copy

    candidate = copy.deepcopy(params)
    candidate["name"] = f"mut-{rng.randrange(10 ** 6):06d}"
    if rng.random() < 0.6:
        candidate["min_solved"] = max(int(round(float(candidate.get("min_solved", 1)))) + rng.choice((-1, 1)), 1)
    if rng.random() < 0.6:
        candidate["score_bar"] = round(min(max(float(candidate.get("score_bar", 0.55)) + rng.uniform(-0.05, 0.05), 0.1), 0.9), 4)
    if rng.random() < 0.4:
        candidate["min_success_rate"] = round(min(max(float(candidate.get("min_success_rate", 0.5)) + rng.uniform(-0.1, 0.1), 0.0), 1.0), 4)
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
                    pool.extend(mutate_params(base, rng) for _ in range(max(POLICY_VERSIONS - 1, 1)))

                ranking = []
                for params in pool:
                    normalized = policy_params(params)
                    result = replay_score(trees, normalized)
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
