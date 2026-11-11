"""Generalize operator (Harmony Harness E.3): audit where muscle SOPs are filed.

Scope is frontmatter, so "moving a unit between scopes" is a metadata rewrite
plus — for the derive case — an LLM-suggested de-hardcoding note. The operator
adds no new capability; it only corrects misfiled scope so future promotions
and syncs compose cleanly across devices. Failures degrade to a no-op report.
"""

import asyncio
import json

import frontmatter

from ...components import R
from ..base_step import BaseStep
from ..evolve._evolve import agent_reply_result_text
from ..evolve.dream.utils import llm_available, parse_structured_reply
from .usage import now_iso


def parse_audit_reply(text: str, known_paths: set[str]) -> list[dict]:
    """Validate the auditor reply into scope moves; unknown paths are dropped."""
    data = parse_structured_reply(text, strict=True)
    raw_moves = data.get("moves") if isinstance(data, dict) else None
    if not isinstance(raw_moves, list):
        return []
    moves: list[dict] = []
    for item in raw_moves:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        scope = str(item.get("scope") or "").strip().lower()
        if path not in known_paths:
            continue
        if action not in {"promote", "demote", "keep"}:
            continue
        if action == "promote" and scope != "general":
            scope = "general"
        if action == "demote":
            device = str(item.get("device") or "").strip()
            if not device:
                continue
            scope = f"device:{device}"
        moves.append({"path": path, "action": action, "scope": scope, "reason": str(item.get("reason") or "")})
    return moves


@R.register("ops_generalize_step")
class OpsGeneralizeStep(BaseStep):
    """Audit muscle SOP scope placement; apply promote/demote moves.

    Without an LLM this is a no-op report. Every failure path degrades to a
    skipped entry — the operator changes filing, never capability.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        root = workspace / muscle_rel
        files = sorted(root.rglob("*.md")) if root.is_dir() else []
        known = {p.relative_to(workspace).as_posix() for p in files if p.is_file()}
        if not known:
            self.context.response.answer = "No muscle SOPs to audit"
            self.context.response.metadata["generalize"] = {"audited": 0, "moves": []}
            return self.context.response
        if not llm_available(self):
            self.context.response.metadata["generalize"] = {
                "audited": len(known),
                "moves": [],
                "skipped": "no llm configured",
            }
            self.context.response.answer = f"Audited {len(known)} SOP(s); generalize skipped (no LLM)"
            return self.context.response

        bodies = {}
        for rel in known:
            post = frontmatter.loads((workspace / rel).read_text(encoding="utf-8", errors="replace"))
            bodies[rel] = {
                "path": rel,
                "scope": str(post.metadata.get("scope") or "general"),
                "promoted_from": str(post.metadata.get("promoted_from") or ""),
                "body": post.content[:2500],
            }

        audit_timeout = float(self.kwargs.get("audit_timeout", 90.0))
        system_prompt = self.prompt_format("generalize_system_prompt")
        user_message = self.prompt_format("generalize_user_message", sops_json=json.dumps(bodies, ensure_ascii=False))
        try:
            result = await asyncio.wait_for(
                self.agent_wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=audit_timeout,
            )
            moves = parse_audit_reply(agent_reply_result_text(result), known)
        except Exception as exc:  # noqa: BLE001 - audit failure degrades to a report
            self.context.response.metadata["generalize"] = {
                "audited": len(known),
                "moves": [],
                "skipped": f"{type(exc).__name__}: {exc}",
            }
            self.context.response.answer = f"Audited {len(known)} SOP(s); generalize failed: {exc}"
            return self.context.response

        applied: list[dict] = []
        for move in moves:
            if move["action"] == "keep":
                applied.append({**move, "applied": False, "reason": move["reason"] or "no change"})
                continue
            await self.run_job("frontmatter_update", path=move["path"],
                               metadata={"scope": move["scope"], "scope_audited_at": now_iso()})
            applied.append({**move, "applied": True})

        moved = [entry for entry in applied if entry.get("applied")]
        self.context.response.answer = (
            f"Generalize audit: {len(known)} SOP(s) audited, {len(moved)} scope move(s) applied"
        )
        self.context.response.metadata["generalize"] = {"audited": len(known), "moves": applied}
        return self.context.response
