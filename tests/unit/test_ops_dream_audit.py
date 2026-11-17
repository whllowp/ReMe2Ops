"""Edge dream audit: candidate selection, closed vocabulary, ledger, regressions."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops import promote as promote_module
from reme.steps.ops import reflex_index as reflex_index_module
from reme.steps.ops import tiered_search as tiered_search_module
from reme.steps.ops.dream_audit import (
    DEFAULT_VERIFIED_STATUS,
    STATUS_UNCONFIRMED,
    VERIFIED_STATUS_VALUES,
    OpsDreamAuditStep,
    audit_state_path,
    audit_status_path,
    body_sha256,
    build_audit_note,
    describe_evidence,
    evidence_paths,
    load_status_ledger,
    parse_audit_reply,
    pick_evidence,
    select_audit_candidates,
)

MUSCLE_A = "digest/muscle/a.md"
MUSCLE_B = "digest/muscle/b.md"


def _write_mirror(tmp_path: Path, rel: str, body: str, metadata: dict | None = None) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {"name": Path(rel).stem}
    path.write_text(frontmatter.dumps(frontmatter.Post(body, **metadata)), encoding="utf-8")


def _append_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_feedback" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _step(tmp_path: Path, **kwargs):
    return OpsDreamAuditStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        **kwargs,
    )


class FakeWrapper(BaseAgentWrapper):
    """One structured verdict batch, then repeated on every call."""

    def __init__(self, verdicts: list[dict]):
        super().__init__()
        self.calls = []
        self.verdicts = verdicts

    async def reply(self, inputs, **kwargs):
        self.calls.append(inputs)
        return {
            "status": "completed",
            "error": None,
            "result": json.dumps({"verdicts": self.verdicts}),
        }


def test_select_audit_candidates_new_evidence_and_hash_mismatch():
    mirrors = {MUSCLE_A: "body-a", MUSCLE_B: "body-b", "digest/muscle/c.md": "body-c"}
    ledger = {
        # a: audited and unchanged -> not a candidate
        MUSCLE_A: {"status": "confirmed", "body_sha256": body_sha256("body-a")},
        # b: hash mismatch -> candidate even without new usage
        MUSCLE_B: {"status": "confirmed", "body_sha256": "stale"},
    }
    records = [
        {"ts": "t0", "path": "digest/muscle/c.md", "outcome": "failed"},  # below cursor: ignored
        {"ts": "t1", "path": MUSCLE_B, "outcome": "solved"},
        {"ts": "t2", "path": "digest/muscle/c.md", "outcome": "failed"},
    ]
    selected = select_audit_candidates(mirrors, ledger, records, cursor=1, max_candidates=10)
    assert selected[0] == "digest/muscle/c.md"  # failure evidence sorts first
    assert set(selected) == {"digest/muscle/c.md", MUSCLE_B}
    capped = select_audit_candidates(mirrors, ledger, records, cursor=1, max_candidates=1)
    assert capped == ["digest/muscle/c.md"]


def test_evidence_paths_is_precap_and_cursor_gated():
    """The deferral set spans truncated candidates and respects the cursor."""
    mirrors = {MUSCLE_A: "body-a", MUSCLE_B: "body-b", "digest/muscle/c.md": "body-c"}
    records = [
        {"ts": "t0", "path": "digest/muscle/c.md", "outcome": "failed"},  # below cursor
        {"ts": "t1", "path": MUSCLE_B, "outcome": "solved"},  # solved is not evidence
        {"ts": "t2", "path": MUSCLE_A, "outcome": "failed"},
        {"ts": "t3", "path": "digest/wiki/not-a-mirror.md", "outcome": "failed"},  # not a mirror
    ]
    assert evidence_paths(mirrors, records, cursor=1) == {MUSCLE_A}
    assert evidence_paths(mirrors, records, cursor=0) == {"digest/muscle/c.md", MUSCLE_A}
    assert evidence_paths(mirrors, records, cursor=99) == set()
    # pre-cap: every evidence-bearing selected candidate is inside the full set
    selected = select_audit_candidates(mirrors, ledger={}, usage_records=records, cursor=1, max_candidates=10)
    assert set(selected) & evidence_paths(mirrors, records, cursor=1) == evidence_paths(mirrors, records, cursor=1)


def test_parse_audit_reply_accepts_only_closed_vocabulary():
    known = {MUSCLE_A}
    reply = json.dumps(
        {
            "verdicts": [
                {"path": MUSCLE_A, "status": "confirmed", "correction": ""},
                {"path": MUSCLE_A, "status": "conflict", "correction": "wrong flag order"},
                {"path": MUSCLE_A, "status": "unconfirmed", "correction": ""},
                {"path": MUSCLE_A, "status": "verified", "correction": "legacy word"},  # rejected
                {"path": "digest/muscle/ghost.md", "status": "confirmed", "correction": ""},  # unknown path
                "junk",
            ],
        },
    )
    verdicts = parse_audit_reply(reply, known)
    assert [v["status"] for v in verdicts] == ["confirmed", "conflict", "unconfirmed"]
    assert verdicts[1]["correction"] == "wrong flag order"


def test_verified_vocabulary_is_shared_across_the_loop():
    import reme.steps.ops.dream_audit as dream_audit

    assert VERIFIED_STATUS_VALUES == ("confirmed", "unconfirmed", "conflict")
    assert tiered_search_module.VERIFIED_STATUS_VALUES is dream_audit.VERIFIED_STATUS_VALUES
    assert reflex_index_module.DEFAULT_VERIFIED_STATUS is dream_audit.DEFAULT_VERIFIED_STATUS
    assert promote_module.DEFAULT_VERIFIED_STATUS in VERIFIED_STATUS_VALUES
    assert DEFAULT_VERIFIED_STATUS == STATUS_UNCONFIRMED


def test_pick_evidence_prefers_failures_then_sessions():
    records = [
        {"ts": "t1", "path": MUSCLE_A, "outcome": "solved", "session_id": "s-old"},
        {"ts": "t2", "path": MUSCLE_A, "outcome": "failed", "query": "disk full"},
    ]
    assert pick_evidence(records, MUSCLE_A) == {"kind": "query", "query": "disk full"}
    failed_session = [
        {"ts": "t1", "path": MUSCLE_A, "outcome": "solved", "session_id": "s-old"},
        {"ts": "t2", "path": MUSCLE_A, "outcome": "failed", "session_id": "s-new", "query": "ignored"},
    ]
    assert pick_evidence(failed_session, MUSCLE_A) == {"kind": "session", "session_id": "s-new"}
    assert pick_evidence(records, "digest/muscle/none.md") is None
    assert describe_evidence({"kind": "session", "session_id": "s1"}) == "session:s1"
    assert describe_evidence(None) == "usage:counters"


def test_build_audit_note_lists_conflicts_with_provenance():
    note = build_audit_note(
        [
            {
                "path": MUSCLE_A,
                "status": "conflict",
                "correction": "flag order is --force first",
                "verified_by": "session:s9",
            },
        ],
    )
    assert f"[[{MUSCLE_A}]]" in note
    assert "flag order is --force first" in note
    assert "session:s9" in note


@pytest.mark.asyncio
async def test_audit_without_llm_is_pure_report(tmp_path):
    _write_mirror(tmp_path, MUSCLE_A, "runbook body")
    _append_usage(tmp_path, [{"ts": "t1", "path": MUSCLE_A, "outcome": "failed", "query": "disk full"}])

    step = _step(tmp_path)
    response = await step(RuntimeContext())

    meta = response.metadata["ops_audit"]
    assert meta["skipped"] == "no llm configured"
    assert meta["candidates"] == [MUSCLE_A]
    assert meta["audited"] == 0
    assert not audit_status_path(tmp_path, "metadata").exists()  # no ledger write
    assert not audit_state_path(tmp_path, "metadata").exists()  # cursor untouched
    post = frontmatter.loads((tmp_path / MUSCLE_A).read_text(encoding="utf-8"))
    assert "verified_status" not in post.metadata  # no stamping
    assert not list((tmp_path / "daily").rglob("ops-audit-*"))  # no correction note


@pytest.mark.asyncio
async def test_audit_with_no_candidates_skips_llm(tmp_path):
    _write_mirror(tmp_path, MUSCLE_A, "runbook body")
    wrapper = FakeWrapper([])
    step = _step(tmp_path, agent_wrapper=wrapper)
    response = await step(RuntimeContext())

    assert "no candidates" in response.answer
    assert wrapper.calls == []  # early exit before the LLM
    assert response.metadata["ops_audit"]["cursor"] == 0


@pytest.mark.asyncio
async def test_audit_stamps_ledger_mirror_and_note(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.dream_audit.llm_available", lambda step: True)
    body = "runbook body v1"
    _write_mirror(tmp_path, MUSCLE_A, body, {"name": "a", "description": "Disk cleanup"})
    _append_usage(
        tmp_path,
        [
            {"ts": "t1", "path": MUSCLE_A, "outcome": "solved", "query": "disk full"},
            {"ts": "t2", "path": MUSCLE_A, "outcome": "failed", "session_id": "sess-1"},
        ],
    )
    wrapper = FakeWrapper(
        [
            {"path": MUSCLE_A, "status": "conflict", "correction": "order is --force first"},
        ],
    )
    step = _step(tmp_path, agent_wrapper=wrapper)
    writes: list[tuple[str, dict]] = []

    async def fake_run_job(name, /, **kwargs):
        writes.append((name, kwargs))
        if name == "session_read":
            response = Mock()
            response.answer = {"messages": [{"role": "user", "content": "disk full; --force failed"}]}
            return response
        if name == "frontmatter_update":
            target = tmp_path / kwargs["path"]
            post = frontmatter.loads(target.read_text(encoding="utf-8"))
            post.metadata.update(kwargs["metadata"])
            target.write_text(frontmatter.dumps(post), encoding="utf-8")
        if name == "write":
            target = tmp_path / kwargs["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = kwargs.get("metadata") or {}
            target.write_text(frontmatter.dumps(frontmatter.Post(kwargs["content"], **metadata)), encoding="utf-8")
        return Mock(success=True, answer="ok")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext(date="2026-11-16"))

    meta = response.metadata["ops_audit"]
    assert meta["audited"] == 1 and meta["cursor"] == 2
    ledger = load_status_ledger(audit_status_path(tmp_path, "metadata"))
    entry = ledger[MUSCLE_A]
    assert entry["status"] == "conflict"
    assert entry["body_sha256"] == body_sha256(body)
    assert entry["verified_by"] == "session:sess-1"  # records how to re-verify
    post = frontmatter.loads((tmp_path / MUSCLE_A).read_text(encoding="utf-8"))
    assert post.metadata["verified_status"] == "conflict"  # decoration stamp
    note = tmp_path / meta["note"]
    assert note.is_file() and note.relative_to(tmp_path).as_posix().startswith("daily/2026-11-16/ops-audit-")
    assert "--force first" in note.read_text(encoding="utf-8")
    state = json.loads(audit_state_path(tmp_path, "metadata").read_text(encoding="utf-8"))
    assert state["audited_count"] == 2  # cursor advanced
    kinds = [name for name, _ in writes]
    assert kinds == ["session_read", "frontmatter_update", "write"]
    assert "--force failed" in wrapper.calls[0]  # transcript evidence reached the prompt


@pytest.mark.asyncio
async def test_hash_mismatch_requeues_even_without_new_usage(tmp_path, monkeypatch):
    """A mirror rewritten upstream (hash mismatch) is re-audited with no new evidence."""
    monkeypatch.setattr("reme.steps.ops.dream_audit.llm_available", lambda step: True)
    _write_mirror(tmp_path, MUSCLE_A, "rewritten body")
    ledger_path = audit_status_path(tmp_path, "metadata")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({MUSCLE_A: {"status": "confirmed", "body_sha256": body_sha256("old body")}}),
        encoding="utf-8",
    )
    state_path = audit_state_path(tmp_path, "metadata")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"audited_count": 1}), encoding="utf-8")  # all usage consumed
    _append_usage(tmp_path, [{"ts": "t1", "path": MUSCLE_A, "outcome": "solved"}])

    wrapper = FakeWrapper([{"path": MUSCLE_A, "status": "confirmed", "correction": ""}])
    step = _step(tmp_path, agent_wrapper=wrapper)
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="ok"))
    response = await step(RuntimeContext())

    meta = response.metadata["ops_audit"]
    assert meta["candidates"] == [MUSCLE_A]  # hash mismatch alone queued it
    ledger = load_status_ledger(ledger_path)
    assert ledger[MUSCLE_A]["status"] == "confirmed"
    assert ledger[MUSCLE_A]["body_sha256"] == body_sha256("rewritten body")


@pytest.mark.asyncio
async def test_audit_llm_failure_degrades_to_report(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.dream_audit.llm_available", lambda step: True)
    _write_mirror(tmp_path, MUSCLE_A, "body")
    _append_usage(tmp_path, [{"ts": "t1", "path": MUSCLE_A, "outcome": "failed", "query": "q"}])

    class ExplodingWrapper(BaseAgentWrapper):
        async def reply(self, inputs, **kwargs):
            raise RuntimeError("boom")

    step = _step(tmp_path, agent_wrapper=ExplodingWrapper())
    response = await step(RuntimeContext())

    assert response.success is True  # degrades, does not fail the job
    assert "Dream audit failed" in response.answer
    meta = response.metadata["ops_audit"]
    assert meta["skipped"].startswith("RuntimeError")
    assert not audit_status_path(tmp_path, "metadata").exists()


@pytest.mark.asyncio
async def test_truncated_candidates_hold_the_cursor(tmp_path, monkeypatch):
    """max_audit_per_run truncation defers evidence instead of consuming it."""
    monkeypatch.setattr("reme.steps.ops.dream_audit.llm_available", lambda step: True)
    _write_mirror(tmp_path, MUSCLE_A, "body-a")
    _write_mirror(tmp_path, MUSCLE_B, "body-b")
    _append_usage(
        tmp_path,
        [
            {"ts": "t1", "path": MUSCLE_A, "outcome": "failed", "query": "q-a"},
            {"ts": "t2", "path": MUSCLE_B, "outcome": "failed", "query": "q-b"},
        ],
    )
    wrapper = FakeWrapper([{"path": MUSCLE_A, "status": "confirmed", "correction": ""}])
    step = _step(tmp_path, agent_wrapper=wrapper, max_audit_per_run=1)
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="ok"))
    response = await step(RuntimeContext())

    meta = response.metadata["ops_audit"]
    assert meta["candidates"] == [MUSCLE_A]  # capped to one
    assert meta["deferred_evidence"] == [MUSCLE_B]  # B was dropped while holding evidence
    assert meta["cursor"] == 0  # held: B's evidence stays unconsumed
    assert "cursor held" in response.answer
    assert not audit_state_path(tmp_path, "metadata").exists()  # cursor never written
    ledger = load_status_ledger(audit_status_path(tmp_path, "metadata"))
    assert set(ledger) == {MUSCLE_A}  # the applied verdict still lands in the ledger

    # Next run with room: the deferred candidate is re-selected and the cursor moves.
    wrapper2 = FakeWrapper(
        [
            {"path": MUSCLE_A, "status": "confirmed", "correction": ""},
            {"path": MUSCLE_B, "status": "confirmed", "correction": ""},
        ],
    )
    step2 = _step(tmp_path, agent_wrapper=wrapper2, max_audit_per_run=10)
    step2.run_job = AsyncMock(return_value=Mock(success=True, answer="ok"))
    response2 = await step2(RuntimeContext())

    meta2 = response2.metadata["ops_audit"]
    assert meta2["deferred_evidence"] == []
    assert meta2["cursor"] == 2  # len(records)
    state = json.loads(audit_state_path(tmp_path, "metadata").read_text(encoding="utf-8"))
    assert state["audited_count"] == 2


@pytest.mark.asyncio
async def test_filtered_verdicts_hold_the_cursor(tmp_path, monkeypatch):
    """A reply whose verdicts all fail validation consumes no evidence."""
    monkeypatch.setattr("reme.steps.ops.dream_audit.llm_available", lambda step: True)
    _write_mirror(tmp_path, MUSCLE_A, "body-a")
    _append_usage(tmp_path, [{"ts": "t1", "path": MUSCLE_A, "outcome": "failed", "query": "q"}])
    # hallucinated path + legacy vocabulary word: parse_audit_reply drops both
    wrapper = FakeWrapper(
        [
            {"path": "digest/muscle/ghost.md", "status": "confirmed", "correction": ""},
            {"path": MUSCLE_A, "status": "verified", "correction": "legacy word"},
        ],
    )
    step = _step(tmp_path, agent_wrapper=wrapper)
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="ok"))
    response = await step(RuntimeContext())

    meta = response.metadata["ops_audit"]
    assert meta["audited"] == 0
    assert meta["cursor"] == 0  # held: the candidate's evidence is not consumed
    assert meta["deferred_evidence"] == [MUSCLE_A]
    assert not audit_status_path(tmp_path, "metadata").exists()  # no ledger entry
    assert not audit_state_path(tmp_path, "metadata").exists()  # cursor never written
    assert not step.run_job.await_args_list or all(
        call.args[0] != "frontmatter_update" for call in step.run_job.await_args_list
    )  # nothing stamped either


@pytest.mark.asyncio
async def test_audit_empty_mirror_is_noop(tmp_path):
    (tmp_path / "digest").mkdir()
    step = _step(tmp_path)
    response = await step(RuntimeContext())
    assert response.answer == "No muscle SOPs to audit"
    assert response.metadata["ops_audit"]["audited"] == 0
