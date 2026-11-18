"""Cloud-side evidence ledger: auditable proof/conflict counters per digest node.

Reference: Hindsight ``Consolidation (Observations)`` — a belief keeps its
supporting evidence as exact citations plus a proof count, and new evidence
strengthens / weakens / extends the belief rather than overwriting it. The ops
analogue: every digest node already carries provenance as ``## Sources``
wikilinks pointing back at the edge daily notes that produced it (design doc
§1), and the cloud usage aggregate records solved/failed outcomes per node.
What was missing is a structured, auditable join of the two: this step derives

    metadata/ops_evidence/ledger.json = {
      "<digest path>": {
        "sources": [...],          # deduped ## Sources wikilink targets
        "proof_count": int,        # how many distinct sources back the node
        "conflict_count": int,     # failed usage outcomes (weaken signal)
        "last_conflict_at": str    # newest failed usage ts ("" when none)
      }, ...
    }

fully deterministically (no LLM) from the digest workspace plus the cloud
usage JSONL, so each run is an idempotent rebuild — the extend-not-overwrite
semantics come from the derivation itself: the ledger always reflects the
current evidence, and nothing is ever invented.

Shared-Response contract: this step runs inside ``ops_maintenance_cron``
after decay/profile/transitions/link_repair, which share one Response and
whose final ``success``/``answer`` feed the job status — so only
``metadata["ops_evidence"]`` is written here.
"""

import json
import re
from pathlib import Path

import frontmatter

from ...components import R
from ..base_step import BaseStep
from .dream_audit import load_json_state
from .usage import aggregate_usage, normalize_outcome, read_usage_records, usage_file_path

EVIDENCE_DIR = "ops_evidence"
EVIDENCE_LEDGER_FILE = "ledger.json"

# The Sources heading is exactly h2 (integrate.yaml prompt convention: `## Sources`).
_SOURCES_SECTION = re.compile(r"^##[ \t]*Sources[ \t]*$", re.M)
_HEADING = re.compile(r"^#{1,6}[ \t]", re.M)
_WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")


def evidence_ledger_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the evidence ledger."""
    return Path(workspace) / metadata_dir / EVIDENCE_DIR / EVIDENCE_LEDGER_FILE


def parse_wikilinks(text: str) -> list[str]:
    """Sorted deduped wikilink targets; alias (``|``) and anchor (``#``) parts dropped."""
    targets: set[str] = set()
    for raw in _WIKILINK.findall(text or ""):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            targets.add(target)
    return sorted(targets)


def extract_sources(body: str) -> list[str]:
    """Sorted deduped wikilink targets across all ``## Sources`` sections."""
    if not body:
        return []
    targets: set[str] = set()
    for match in _SOURCES_SECTION.finditer(body):
        start = match.end()
        nxt = _HEADING.search(body, start)
        end = nxt.start() if nxt else len(body)
        targets.update(parse_wikilinks(body[start:end]))
    return sorted(targets)


def conflict_stats(records: list[dict]) -> dict[str, dict]:
    """Per-path failed-outcome aggregation — reuse ``aggregate_usage`` on failed records only.

    Returns ``{path: {failed, last_used, ...}}``; ``failed`` is the conflict
    count and ``last_used`` the newest failed ts (the weaken signal).
    """
    failed_only = [record for record in records if isinstance(record, dict) and normalize_outcome(record.get("outcome")) == "failed"]
    return aggregate_usage(failed_only)


def build_ledger_entries(bodies: dict[str, str], conflicts: dict[str, dict]) -> dict[str, dict]:
    """Ledger entries for every scanned digest node (no-Sources nodes count zero)."""
    ledger: dict[str, dict] = {}
    for path in sorted(bodies):
        sources = extract_sources(bodies[path])
        conflict = conflicts.get(path) or {}
        ledger[path] = {
            "sources": sources,
            "proof_count": len(sources),
            "conflict_count": int(conflict.get("failed") or 0),
            "last_conflict_at": str(conflict.get("last_used") or ""),
        }
    return ledger


def load_evidence_ledger(path: Path) -> dict:
    """Read the evidence ledger; tolerate a missing or corrupted file."""
    return load_json_state(path)


def save_evidence_ledger(path: Path, ledger: dict) -> None:
    """Persist the evidence ledger (singleton cloud writer; sort_keys keeps reruns byte-identical)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_node_body(path: Path) -> str | None:
    """Frontmatter-stripped body of one digest node; ``None`` when unreadable."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return frontmatter.loads(raw).content
    except Exception:  # noqa: BLE001 - unparsable frontmatter still yields the raw body
        return raw


@R.register("ops_evidence_step")
class OpsEvidenceStep(BaseStep):
    """Build the evidence ledger for every digest node.

    Sources come from each node's ``## Sources`` wikilinks (frontmatter
    stripped); conflicts come from the cloud usage aggregate
    (``metadata/ops_usage/usage.jsonl``, failed outcomes only). The ledger is
    rewritten wholesale per run — deterministic, idempotent, corruption
    tolerant on the read side. Job success accounting stays with the cron
    chain: only ``metadata["ops_evidence"]`` is written.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path.resolve()
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")

        root = workspace / digest_dir
        bodies: dict[str, str] = {}
        for file_path in sorted(root.rglob("*.md")) if root.is_dir() else []:
            body = read_node_body(file_path)
            if body is not None:
                bodies[file_path.relative_to(workspace).as_posix()] = body

        records = read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True))
        ledger = build_ledger_entries(bodies, conflict_stats(records))

        ledger_path = evidence_ledger_path(workspace, metadata_dir)
        save_evidence_ledger(ledger_path, ledger)

        self.context.response.metadata["ops_evidence"] = {
            "nodes": len(ledger),
            "ledger": ledger_path.relative_to(workspace).as_posix(),
            "proof_total": sum(entry["proof_count"] for entry in ledger.values()),
            "conflict_total": sum(entry["conflict_count"] for entry in ledger.values()),
            "conflict_nodes": sorted(path for path, entry in ledger.items() if entry["conflict_count"]),
            "usage_records": len(records),
        }
        return self.context.response
