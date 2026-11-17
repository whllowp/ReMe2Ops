"""Edge-side dream audit: re-check muscle mirrors against session evidence.

Paper components ① (Mt. Moon dreaming correction: rewrite notes against the
transcript) and ④ (salience: only keep rules/lessons the environment cannot
reveal at run time). The auditor compares candidate muscle mirrors with the
usage evidence that produced them — the session transcript via ``session_read``
when a session id was recorded, else the query text — and stamps exactly one
status from a single closed vocabulary::

    confirmed | unconfirmed | conflict

The weighting authority is the status ledger (``metadata/ops_audit/status.json``)
keyed by path and gated by ``body_sha256``, the hash of the frontmatter-stripped
body. Stamping the mirror frontmatter is operations-visible decoration only:
``ops_muscle_sync`` rewrites mirrors wholesale on remote mtime change, which
wipes stamps but not the ledger — and the resulting hash mismatch re-queues the
file for audit even without new usage evidence. Correction notes for conflicts
are written into ``daily/`` so the nightly ``ops_dream_push`` fixes the cloud
long-term copy through the existing CORRECT ingest semantics. Without an LLM
the step degrades to a pure report (same contract as ``ops_generalize_step``).
"""

import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path

import frontmatter

from ...components import R
from ..base_step import BaseStep
from ..evolve._evolve import agent_reply_result_text
from ..evolve.dream.utils import llm_available, parse_structured_reply, today
from .usage import now_iso, read_usage_records, usage_file_path

# The single verified-status vocabulary for the whole learning loop: the audit
# step stamps it, promotion defaults to it, search weights read it. Keep the
# tuple as the one definition; every other module imports this constant.
VERIFIED_STATUS_VALUES = ("confirmed", "unconfirmed", "conflict")
STATUS_CONFIRMED = "confirmed"
STATUS_UNCONFIRMED = "unconfirmed"
STATUS_CONFLICT = "conflict"
DEFAULT_VERIFIED_STATUS = STATUS_UNCONFIRMED

AUDIT_STATUS_FILE = "status.json"
AUDIT_STATE_FILE = "state.json"

_BODY_EXCERPT_CHARS = 2500


def body_sha256(body: str) -> str:
    """Hash the frontmatter-stripped body; stamping metadata never changes it."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_body_text(path: Path) -> str | None:
    """Frontmatter-stripped body of one markdown file; None when unreadable."""
    try:
        return frontmatter.loads(path.read_text(encoding="utf-8", errors="replace")).content
    except Exception:  # noqa: BLE001 - unreadable mirrors simply carry no status
        return None


def audit_status_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the verified-status ledger (the weighting authority)."""
    return Path(workspace) / metadata_dir / "ops_audit" / AUDIT_STATUS_FILE


def audit_state_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the audit cursor state (usage records consumed)."""
    return Path(workspace) / metadata_dir / "ops_audit" / AUDIT_STATE_FILE


def load_json_state(path: Path) -> dict:
    """Tolerant JSON loader shared by the ledger and the cursor state."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_status_ledger(path: Path) -> dict:
    """Read the status ledger; tolerate a missing or corrupted file."""
    return load_json_state(path)


def save_status_ledger(path: Path, ledger: dict) -> None:
    """Persist the status ledger (singleton edge writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_audit_state(path: Path) -> dict:
    """Read the audit cursor state; tolerate a missing or corrupted file."""
    return load_json_state(path)


def save_audit_state(path: Path, state: dict) -> None:
    """Persist the audit cursor state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def ledger_status_for(path: str, body: str, ledger: dict) -> str | None:
    """Ledger status for a path, but only while the recorded body hash matches.

    A mirror rewritten by ``ops_muscle_sync`` (or edited by hand) no longer
    matches the audited hash: its status drops out and callers treat the file
    as unconfirmed/neutral instead of weighting stale verdicts.
    """
    entry = ledger.get(path) if isinstance(ledger, dict) else None
    if not isinstance(entry, dict):
        return None
    status = str(entry.get("status") or "").strip().lower()
    if status not in VERIFIED_STATUS_VALUES:
        return None
    if entry.get("body_sha256") != body_sha256(body):
        return None
    return status


def evidence_paths(mirrors: dict[str, str], usage_records: list[dict], cursor: int) -> set[str]:
    """Mirror paths with failed usage evidence after the cursor, before any cap.

    This is the full set whose consumption the cursor must account for —
    paths truncated out of the candidate list still belong here.
    """
    paths: set[str] = set()
    for index, record in enumerate(usage_records):
        if index < cursor:
            continue
        path = str(record.get("path") or "").strip()
        if path in mirrors and str(record.get("outcome") or "") == "failed":
            paths.add(path)
    return paths


def select_audit_candidates(
    mirrors: dict[str, str],
    ledger: dict,
    usage_records: list[dict],
    cursor: int,
    *,
    max_candidates: int = 6,
) -> list[str]:
    """Paths worth re-auditing, failures first, deterministically capped.

    A path qualifies when usage evidence newer than ``cursor`` mentions it, or
    when its current body hash no longer matches the ledger (an upstream
    correction rewrote the mirror — exactly the files most worth re-reading).
    """
    new_failed: dict[str, int] = {}
    for index, record in enumerate(usage_records):
        if index < cursor:
            continue
        path = str(record.get("path") or "").strip()
        if path in mirrors and str(record.get("outcome") or "") == "failed":
            new_failed[path] = new_failed.get(path, 0) + 1
    candidates = set(new_failed)
    for path, body in mirrors.items():
        entry = ledger.get(path)
        if isinstance(entry, dict) and entry.get("body_sha256") != body_sha256(body):
            candidates.add(path)
    ordered = sorted(candidates, key=lambda p: (-new_failed.get(p, 0), p))
    return ordered[: max(int(max_candidates), 1)]


def pick_evidence(records: list[dict], path: str) -> dict | None:
    """Best ground-truth pointer for one path: newest failures first, then any
    newest record; a session id beats a bare query because transcripts show
    what actually happened."""
    path_records = [record for record in records if str(record.get("path") or "").strip() == path]
    if not path_records:
        return None
    failed = [record for record in path_records if str(record.get("outcome") or "") == "failed"]
    for pool in (failed, path_records):
        for record in pool:
            session_id = str(record.get("session_id") or "").strip()
            if session_id:
                return {"kind": "session", "session_id": session_id}
        for record in pool:
            query = str(record.get("query") or "").strip()
            if query:
                return {"kind": "query", "query": query}
    return {"kind": "usage", "outcome": str(path_records[-1].get("outcome") or "")}


def describe_evidence(evidence: dict | None) -> str:
    """One-line provenance of how a verdict can be re-checked (ledger field)."""
    if not evidence:
        return "usage:counters"
    kind = evidence.get("kind")
    if kind == "session":
        return f"session:{evidence.get('session_id')}"
    if kind == "query":
        return f"query:{evidence.get('query')}"
    return f"usage:{evidence.get('outcome') or 'unknown'}"


def flatten_session_answer(answer, *, max_chars: int = 4000) -> str:
    """Flatten a ``session_read`` result (SessionSnapshot dump or plain text)
    into bounded transcript evidence."""
    if isinstance(answer, dict):
        lines = []
        for message in answer.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"{message.get('role') or '?'}: {content}")
        text = "\n".join(lines)
    else:
        text = str(answer or "")
    return text[: max(int(max_chars), 200)]


def parse_audit_reply(text: str, known_paths: set[str]) -> list[dict]:
    """Validate the auditor reply into verdicts; the vocabulary is closed.

    Only ``VERIFIED_STATUS_VALUES`` are accepted — legacy words such as
    ``verified`` are dropped rather than coerced, so the audit stamp, the
    promotion default, and the search weights can never drift apart.
    """
    data = parse_structured_reply(text, strict=True)
    raw = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    verdicts: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if path not in known_paths:
            continue
        if status not in VERIFIED_STATUS_VALUES:
            continue
        verdicts.append({"path": path, "status": status, "correction": str(item.get("correction") or "")})
    return verdicts


def build_audit_note(entries: list[dict]) -> str:
    """Markdown correction note for conflict verdicts (``daily/`` memory).

    The nightly ``ops_dream_push`` picks the note up as a changed daily file
    and the cloud applies the correction to the long-term copy with the
    existing CORRECT ingest semantics.
    """
    lines = [
        "# Ops dream audit corrections",
        "",
        "Muscle SOPs that contradict recorded usage evidence; each line is a",
        "correction to integrate into the long-term digest.",
        "",
    ]
    for entry in entries:
        lines.append(f"- [[{entry['path']}]] — {entry['correction'] or 'conflicts with recorded evidence'}")
        lines.append(f"  - evidence: {entry['verified_by']} (status: {entry['status']})")
    return "\n".join(lines).strip() + "\n"


@R.register("ops_dream_audit_step")
class OpsDreamAuditStep(BaseStep):
    """Audit muscle mirrors against transcripts/usage and stamp the ledger.

    Candidates are usage evidence after the cursor plus hash-mismatched
    mirrors, failures first. The cursor advances only when every
    evidence-bearing candidate received a verdict: candidates lost to the
    ``max_audit_per_run`` cap or to reply filtering keep their evidence and
    are re-selected on the next run. Without an LLM this is a pure report:
    no stamps, no ledger writes, no correction note.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path.resolve()
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        daily = str(self.config_value("daily_dir")).strip("/")
        max_candidates = max(int(self.kwargs.get("max_audit_per_run", 6)), 1)
        audit_timeout = float(self.kwargs.get("audit_timeout", 90.0))
        evidence_chars = max(int(self.kwargs.get("evidence_chars", 4000)), 200)

        root = workspace / muscle_rel
        mirrors: dict[str, str] = {}
        for file_path in sorted(root.rglob("*.md")) if root.is_dir() else []:
            body = read_body_text(file_path)
            if body is not None and file_path.is_file():
                mirrors[file_path.relative_to(workspace).as_posix()] = body
        if not mirrors:
            self.context.response.answer = "No muscle SOPs to audit"
            self.context.response.metadata["ops_audit"] = {"audited": 0, "candidates": [], "cursor": 0}
            return self.context.response

        ledger_path = audit_status_path(workspace, metadata_dir)
        state_path = audit_state_path(workspace, metadata_dir)
        ledger = load_status_ledger(ledger_path)
        state = load_audit_state(state_path)
        cursor = max(int(state.get("audited_count") or 0), 0)
        records = read_usage_records(usage_file_path(workspace, metadata_dir))
        candidates = select_audit_candidates(mirrors, ledger, records, cursor, max_candidates=max_candidates)
        if not candidates:
            self.context.response.answer = (
                f"Dream audit: {len(mirrors)} mirror(s), cursor {cursor}, no candidates"
            )
            self.context.response.metadata["ops_audit"] = {
                "audited": 0,
                "candidates": [],
                "cursor": cursor,
                "mirrors": len(mirrors),
            }
            return self.context.response

        if not llm_available(self):
            self.context.response.metadata["ops_audit"] = {
                "audited": 0,
                "candidates": candidates,
                "cursor": cursor,
                "mirrors": len(mirrors),
                "skipped": "no llm configured",
            }
            self.context.response.answer = (
                f"Dream audit: {len(candidates)} candidate(s); audit skipped (no LLM)"
            )
            return self.context.response

        new_records = records[cursor:]
        evidence = {path: pick_evidence(new_records, path) for path in candidates}
        evidence_refs = {path: describe_evidence(evidence[path]) for path in candidates}

        prompt_bodies = []
        for path in candidates:
            pointer = evidence[path]
            if pointer and pointer["kind"] == "session":
                try:
                    session_response = await self.run_job("session_read", session_id=pointer["session_id"])
                    ground_truth = flatten_session_answer(session_response.answer, max_chars=evidence_chars)
                except Exception as exc:  # noqa: BLE001 - missing transcript still audits on counters
                    ground_truth = f"(session read failed: {type(exc).__name__}: {exc})"
            elif pointer and pointer["kind"] == "query":
                ground_truth = pointer["query"]
            else:
                ground_truth = "usage counters only (no transcript or query recorded)"
            entry = ledger.get(path) if isinstance(ledger.get(path), dict) else {}
            prompt_bodies.append(
                {
                    "path": path,
                    "current_status": str(entry.get("status") or STATUS_UNCONFIRMED),
                    "evidence": evidence_refs[path],
                    "ground_truth": ground_truth,
                    "body": mirrors[path][:_BODY_EXCERPT_CHARS],
                },
            )

        system_prompt = self.prompt_format("dream_audit_system_prompt")
        user_message = self.prompt_format(
            "dream_audit_user_message", candidates_json=json.dumps(prompt_bodies, ensure_ascii=False),
        )
        try:
            result = await asyncio.wait_for(
                self.agent_wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=audit_timeout,
            )
            verdicts = parse_audit_reply(agent_reply_result_text(result), set(candidates))
        except Exception as exc:  # noqa: BLE001 - audit failure degrades to a report
            self.context.response.metadata["ops_audit"] = {
                "audited": 0,
                "candidates": candidates,
                "cursor": cursor,
                "skipped": f"{type(exc).__name__}: {exc}",
            }
            self.context.response.answer = f"Dream audit failed: {exc}"
            return self.context.response

        stamped: list[str] = []
        conflicts: list[dict] = []
        for verdict in verdicts:
            path = verdict["path"]
            entry = {
                "status": verdict["status"],
                "verified_at": now_iso(),
                "verified_by": evidence_refs[path],
                "correction": verdict["correction"],
                "body_sha256": body_sha256(mirrors[path]),
            }
            ledger[path] = entry
            # Mirror frontmatter is operations-visible decoration only; the
            # ledger above is the weighting authority (muscle_sync rewrites
            # mirrors wholesale and would otherwise erase the verdict).
            await self.run_job(
                "frontmatter_update",
                path=path,
                metadata={"verified_status": verdict["status"], "verified_at": entry["verified_at"]},
            )
            stamped.append(path)
            if verdict["status"] == STATUS_CONFLICT and verdict["correction"]:
                conflicts.append({**verdict, "verified_by": entry["verified_by"]})

        note_path = ""
        if conflicts:
            day = today(self, str(self.context.get("date") or ""))
            stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            note_path = f"{daily}/{day}/ops-audit-{stamp}.md"
            await self.run_job(
                "write",
                path=note_path,
                name=f"ops-audit-{stamp}",
                description=f"Dream audit corrections ({len(conflicts)} conflict(s))",
                content=build_audit_note(conflicts),
                metadata={
                    "source": "ops_dream_audit",
                    "audit_cursor": len(records),
                    "conflict_paths": [entry["path"] for entry in conflicts],
                },
            )

        # The cursor only moves past evidence the audit actually consumed.
        # When truncation (max_audit_per_run) or reply filtering dropped a
        # candidate that still has usage evidence after the cursor, that
        # evidence must stay unconsumed — otherwise the candidate would never
        # be re-selected (no new failure, no hash change) and audit coverage
        # would be silently lost. A held cursor costs one bounded retry per
        # run; silent loss is not recoverable. ``evidence_paths`` is computed
        # pre-cap, so candidates cut by truncation are covered too.
        covered = {verdict["path"] for verdict in verdicts}
        deferred = sorted(p for p in evidence_paths(mirrors, records, cursor) if p not in covered)
        if deferred:
            if stamped:
                save_status_ledger(ledger_path, ledger)  # keep the verdicts that did apply
        else:
            state["audited_count"] = len(records)
            save_status_ledger(ledger_path, ledger)
            save_audit_state(state_path, state)

        answer = (
            f"Dream audit: {len(candidates)} candidate(s), {len(verdicts)} verdict(s), "
            f"{len(stamped)} stamped, {len(conflicts)} correction note(s)"
        )
        if deferred:
            answer += f"; cursor held, {len(deferred)} candidate(s) deferred for re-audit"
        self.context.response.answer = answer
        self.context.response.metadata["ops_audit"] = {
            "audited": len(verdicts),
            "candidates": candidates,
            "verdicts": verdicts,
            "stamped": stamped,
            "note": note_path or None,
            "cursor": cursor if deferred else len(records),
            "deferred_evidence": deferred,
            "ledger": ledger_path.relative_to(workspace).as_posix(),
        }
        return self.context.response
