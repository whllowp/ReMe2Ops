"""Cloud-side source reverse index: source path → citing digest nodes.

Reference: Sage-Taiyi (arXiv 2611.12762v2 §Evidence State) — the evidence chain
links sources and conditions, and new evidence revisits only the affected
judgments. Targeted revision therefore needs the reverse of the round-2
evidence ledger: the ledger answers "which sources back THIS node" (forward
view, per node); the index answers "which nodes cite THIS source" (reverse
view, per source). When a daily note is corrected or invalidated,
:func:`affected_nodes` names exactly the digest judgments to revisit — and
:func:`shared_source_nodes` marks the sources cited by two or more nodes,
the shared-premise warning: their agreement is not independent support.

Dangling targets under ``daily/`` are intentional short-term → long-term
provenance virtual nodes (design doc §1) and are indexed as-is, never
repaired or dropped. Parsing reuses the round-2 evidence parsers directly
(``evidence.extract_sources`` / ``evidence.read_node_body``) instead of the
``get_nodes`` graph, so the index reads files the same way the ledger does.

Shared-Response contract: this step runs inside ``ops_maintenance_cron``
after decay/profile/transitions/link_repair/evidence, which share one
Response and whose final ``success``/``answer`` feed the job status — so only
``metadata["source_index"]`` is written here.
"""

import json
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .dream_audit import load_json_state
from .evidence import extract_sources, read_node_body

SOURCE_INDEX_FILE = "ops_source_index.json"
DEFAULT_SHARED_MIN_NODES = 2


def source_index_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the source reverse index."""
    return Path(workspace) / metadata_dir / SOURCE_INDEX_FILE


def build_source_index(bodies: dict[str, str]) -> dict[str, list[str]]:
    """``{source_path: sorted [citing node paths]}`` from ## Sources wikilinks.

    Dangling targets (``daily/`` virtual provenance nodes) are indexed as-is.
    Nodes without a ## Sources section contribute nothing; a node citing one
    source repeatedly counts once (deduped by the shared parser).
    """
    index: dict[str, set[str]] = {}
    for node, body in bodies.items():
        for source in extract_sources(body):
            index.setdefault(source, set()).add(node)
    return {source: sorted(nodes) for source, nodes in sorted(index.items())}


def shared_source_nodes(index: dict[str, list[str]], *, min_nodes: int = DEFAULT_SHARED_MIN_NODES) -> dict[str, list[str]]:
    """Sources cited by at least ``min_nodes`` (default 2) distinct nodes.

    These are the shared-premise entries: nodes agreeing through one source do
    not independently corroborate each other (Taiyi), so targeted revision of
    that source must revisit every node listed here.
    """
    floor = max(int(min_nodes), DEFAULT_SHARED_MIN_NODES)
    return {source: nodes for source, nodes in sorted(index.items()) if len(nodes) >= floor}


def affected_nodes(index: dict[str, list[str]], source_path: str) -> list[str]:
    """Digest nodes whose provenance cites one source — the targeted-revision set."""
    return list(index.get(str(source_path or "").strip(), []))


def load_source_index(path: Path) -> dict:
    """Read the source index; tolerate a missing or corrupted file."""
    return load_json_state(path)


def save_source_index(path: Path, index: dict) -> None:
    """Persist the source index (singleton cloud writer; sort_keys keeps reruns byte-identical)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@R.register("ops_source_index_step")
class OpsSourceIndexStep(BaseStep):
    """Build the source reverse index for targeted revision and relevance marks.

    Scans every digest node's ## Sources wikilinks (files read directly, not
    the ``get_nodes`` graph), writes ``metadata/ops_source_index.json`` as
    ``{source_path: [citing nodes]}``, and reports the shared sources (>= 2
    citing nodes) in ``metadata["source_index"]["shared"]``. Deterministic,
    idempotent rebuild, corruption tolerant on the read side.
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

        index = build_source_index(bodies)
        shared = shared_source_nodes(index)

        index_path = source_index_path(workspace, metadata_dir)
        save_source_index(index_path, index)

        self.context.response.metadata["source_index"] = {
            "sources": len(index),
            "index": index_path.relative_to(workspace).as_posix(),
            "links": sum(len(nodes) for nodes in index.values()),
            "shared": shared,
            "shared_count": len(shared),
        }
        return self.context.response
