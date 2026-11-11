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


@R.register("ops_replay_step")
class OpsReplayStep(BaseStep):
    """Score candidate promotion policies against recorded usage worlds."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        usage_path = usage_file_path(workspace, metadata_dir, cloud=True)
        records = read_usage_records(usage_path)
        trees = build_session_trees(records)
        if not trees:
            self.context.response.success = False
            self.context.response.answer = "No replay worlds: usage history has no session/query grouping"
            return self.context.response

        raw = self.context.get("policies") or self.kwargs.get("policies") or []
        policies = [policy_params(item) for item in raw] if isinstance(raw, list) and raw else default_candidate_grid(
            load_deployed_policy_file(workspace, metadata_dir).get("params"),
        )

        ranking = []
        for params in policies:
            result = replay_score(trees, params)
            ranking.append({"name": params["name"], "params": params, **result})
        ranking.sort(key=lambda item: item["score"], reverse=True)

        deployed = load_deployed_policy_file(workspace, metadata_dir)
        deployed_score = float(deployed.get("replay_score") or float("-inf"))
        winner = ranking[0]
        redeploy = winner["score"] > deployed_score
        self.context["replay_winner"] = winner
        self.context["replay_redeploy"] = redeploy

        self.context.response.answer = (
            f"Replayed {len(policies)} policy/policies over {len(trees)} world(s); "
            f"best={winner['name']} score={winner['score']}"
            + ("" if redeploy else "; deployed policy still best")
        )
        self.context.response.metadata["replay"] = {
            "worlds": len(trees),
            "ranking": ranking,
            "deployed_score": deployed.get("replay_score"),
            "redeploy": redeploy,
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
