"""Decision-model fast track: judge SOPs into muscle memory immediately.

The usage loop (solved>=N, nightly cron) is the slow path to muscle memory.
This module is the fast path: a probabilistic decision model weighs each SOP's
content and usage evidence, and a confident verdict promotes it on the spot.
Judgment is failure-safe by design — a missing model, timeout, or refusal
never blocks promotion or ingestion; those fall back to the usage loop.
"""

import asyncio

from ...components.as_decision_model.base_as_decision_model import BaseAsDecisionModel
from ...components import R
from ...enumeration import ComponentEnum
from ..base_step import BaseStep, Ref
from .usage import aggregate_usage, read_usage_records, usage_file_path

_MUSCLE_QUESTION = {
    "type": "binary",
    "instructions": (
        "Judge whether this ops runbook qualifies as muscle memory: stable and "
        "deterministic steps, broadly reusable beyond one incident, safe to "
        "execute on reflex. Return a low probability for one-off fixes, "
        "environment-specific hacks, incomplete steps, or risky procedures."
    ),
    "criteria": {
        "true": "conditioned-reflex grade: a future incident of this kind should follow it verbatim",
        "false": "stay long-term knowledge; keep promoting only through accumulated usage",
    },
}
_EVIDENCE_QUESTION = {
    "type": "binary",
    "instructions": (
        "Judge whether the recorded usage evidence (solved/failed counts and "
        "success rate) supports making this runbook a conditioned reflex now."
    ),
    "criteria": {
        "true": "the evidence shows reliable real-world success",
        "false": "the evidence is too thin or too flaky to skip the slow usage loop",
    },
}

_BODY_CHARS = 4000


def answer_probability(answer) -> float | None:
    """Extract a finite [0, 1] probability from a classifier answer."""
    value = getattr(answer, "probability", None)
    if (
        getattr(answer, "type", None) == "binary_answer"
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= 1
    ):
        return float(value)
    return None


async def judge_sops(
    step: BaseStep,
    paths: list[str],
    *,
    muscle_worthy_threshold: float = 0.8,
    evidence_threshold: float = 0.6,
    min_solved: int = 1,
    body_chars: int = _BODY_CHARS,
    judge_timeout: float = 30.0,
) -> dict[str, dict]:
    """Judge candidate SOPs with the step's decision model; never raises.

    Returns ``{path: verdict}`` where a verdict is
    ``{judged, muscle_worthy, evidence, solved, success_rate, promotable}`` or
    ``{judged: False, reason}``. ``promotable`` gates on both probabilities and
    ``min_solved`` so a confident-but-unused SOP still waits for real use.
    """
    workspace = step.file_store.workspace_path
    metadata_dir = str(step.config_value("metadata_dir"))
    usage_stats = aggregate_usage(read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True)))

    decision = getattr(step, "as_decision_model", None)
    if decision is None:
        return {path: {"judged": False, "reason": "no decision model configured"} for path in paths}

    entries: dict[str, dict] = {}
    questions: dict[str, dict] = {}
    for index, path in enumerate(paths):
        file_path = workspace / path
        try:
            inside = file_path.resolve().is_relative_to(workspace.resolve())
        except (OSError, ValueError):
            inside = False
        if not inside or not file_path.is_file():
            entries[path] = {"judged": False, "reason": "file missing"}
            continue
        stats = usage_stats.get(path) or {}
        entries[path] = {
            "judged": False,
            "muscle_worthy": None,
            "evidence": None,
            "solved": int(stats.get("solved") or 0),
            "failed": int(stats.get("failed") or 0),
            "success_rate": stats.get("success_rate", 0.0),
        }
        body = file_path.read_text(encoding="utf-8", errors="replace")[:body_chars]
        state_entry = {
            "path": path,
            "body": body,
            "usage": {
                "solved": entries[path]["solved"],
                "failed": entries[path]["failed"],
                "success_rate": entries[path]["success_rate"],
            },
        }
        questions[f"sop{index}_muscle_worthy"] = dict(_MUSCLE_QUESTION)
        questions[f"sop{index}_evidence_sufficient"] = dict(_EVIDENCE_QUESTION)
        entries[path]["_state"] = state_entry

    state = {"candidates": [entry["_state"] for entry in entries.values() if "_state" in entry]}
    try:
        response = await asyncio.wait_for(decision(state=state, questions=questions), timeout=judge_timeout)
    except Exception as exc:  # noqa: BLE001 - a decision failure is never fatal
        reason = f"{type(exc).__name__}: {exc}"
        for entry in entries.values():
            if entry.get("judged") is False and "reason" not in entry:
                entry["reason"] = reason
                entry.pop("_state", None)
        return {path: {key: value for key, value in entry.items() if key != "_state"} for path, entry in entries.items()}

    for index, path in enumerate(paths):
        entry = entries.get(path)
        if entry is None or "_state" not in entry:
            continue
        muscle = answer_probability(response.content.get(f"sop{index}_muscle_worthy"))
        evidence = answer_probability(response.content.get(f"sop{index}_evidence_sufficient"))
        if muscle is None or evidence is None:
            entry["reason"] = "invalid or missing answer"
            continue
        entry["judged"] = True
        entry["muscle_worthy"] = muscle
        entry["evidence"] = evidence
        entry["promotable"] = (
            muscle >= muscle_worthy_threshold
            and evidence >= evidence_threshold
            and entry["solved"] >= min_solved
        )
    return {path: {key: value for key, value in entry.items() if key != "_state"} for path, entry in entries.items()}


@R.register("ops_muscle_judge_step")
class OpsMuscleJudgeStep(BaseStep):
    """Judge SOPs with the decision model and optionally promote them at once.

    Reads explicit ``paths`` or the ``fast_track_candidates`` list emitted by
    ``ops_usage_ingest_step``; with no decision model configured this is a
    no-op that reports the skip reason.
    """

    as_decision_model = Ref(BaseAsDecisionModel, ComponentEnum.AS_DECISION_MODEL, optional=True)

    async def execute(self):
        assert self.context is not None
        raw_paths = (
            self.context.get("paths")
            or self.kwargs.get("paths")
            or self.context.get("fast_track_candidates")
            or []
        )
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        paths = list(dict.fromkeys(paths))
        if not paths:
            self.context.response.answer = "No candidates to judge"
            self.context.response.metadata["judge"] = {"paths": [], "promoted": [], "verdicts": {}}
            return self.context.response

        auto_promote = bool(self.context.get("auto_promote", self.kwargs.get("auto_promote", True)))
        muscle_worthy_threshold = float(self.kwargs.get("muscle_worthy_threshold", 0.8))
        evidence_threshold = float(self.kwargs.get("evidence_threshold", 0.6))
        min_solved = max(int(self.kwargs.get("min_solved", 1)), 1)
        judge_timeout = float(self.kwargs.get("judge_timeout", 30.0))

        verdicts = await judge_sops(
            self,
            paths,
            muscle_worthy_threshold=muscle_worthy_threshold,
            evidence_threshold=evidence_threshold,
            min_solved=min_solved,
            judge_timeout=judge_timeout,
        )

        promoted: list[dict] = []
        if auto_promote:
            for path, verdict in verdicts.items():
                if not verdict.get("promotable"):
                    continue
                response = await self.run_job(
                    "ops_muscle_promote",
                    path=path,
                    judge_probability=verdict.get("muscle_worthy"),
                )
                if response.success:
                    promoted.append({"path": path, "muscle_worthy": verdict.get("muscle_worthy")})
                else:
                    verdict["promote_error"] = str(response.answer)

        judged = sum(1 for verdict in verdicts.values() if verdict.get("judged"))
        self.context.response.answer = (
            f"Judged {judged}/{len(paths)} candidate(s); promoted {len(promoted)} to muscle"
        )
        self.context.response.metadata["judge"] = {
            "paths": paths,
            "judged": judged,
            "promoted": promoted,
            "verdicts": verdicts,
        }
        return self.context.response
