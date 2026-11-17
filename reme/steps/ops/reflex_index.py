"""Bounded MEMORY.md-style reflex index over the muscle mirror.

Paper component ② (promotion + MEMORY.md): every session injects only the head
of this index — a bounded table of contents showing what the edge knows and
where to find it, while the details sink into the linked files. The injection
layer must stay small: it spends tokens in every session and any mistake
propagates into every session, so the index is hard-capped
(:data:`INDEX_MAX_LINES` / :data:`INDEX_MAX_BYTES`), rendered deterministically
(no LLM), and rebuilt from the mirror plus the status ledger at any time
(derived state is rebuildable).

Shared-Response contract: the muscle-sync job shares one RuntimeContext and one
Response across steps and records job success from that final response — so
this step writes only ``metadata["reflex_index"]`` and never touches
``answer``/``success``, keeping the sync answer and failure state intact.
"""

import frontmatter

from ...components import R
from ..base_step import BaseStep
from .dream_audit import (
    DEFAULT_VERIFIED_STATUS,
    STATUS_UNCONFIRMED,
    VERIFIED_STATUS_VALUES,
    audit_status_path,
    ledger_status_for,
    load_status_ledger,
)

INDEX_MAX_LINES = 200
INDEX_MAX_BYTES = 25600
DEFAULT_DAILY_POINTERS = 5
TRUNCATION_NOTICE = "- … index truncated; use ops_search for anything not listed"

_HEADER_LINES = [
    "# Memory Index",
    "",
    "Bounded reflex index rebuilt from the muscle mirror after every sync.",
    "Sessions inject this head only; details live in each linked file below.",
    "",
    "## Muscle reflexes",
    "",
]
_DAILY_HEADING = ["", "## Recent daily notes", ""]
_FOOTER_LINES = ["", "More: run ops_search with the query instead of loading files."]
_MIN_RENDER_LINES = len(_HEADER_LINES) + 2  # header + one entry + footer


def index_entries(mirrors: dict[str, dict], ledger: dict) -> list[dict]:
    """Muscle mirror entries with hash-gated ledger status, sorted by path.

    ``mirrors`` maps path → ``{description, scope, body}``. The status comes
    from the audit ledger and only while the recorded ``body_sha256`` still
    matches the mirror body; anything else renders as ``unconfirmed``.
    """
    entries: list[dict] = []
    for path in sorted(mirrors):
        info = mirrors[path] or {}
        status = ledger_status_for(path, str(info.get("body") or ""), ledger)
        entries.append(
            {
                "path": path,
                "description": str(info.get("description") or ""),
                "scope": str(info.get("scope") or "general"),
                "verified_status": status if status in VERIFIED_STATUS_VALUES else STATUS_UNCONFIRMED,
            },
        )
    return entries


def recent_daily_pointers(paths: list[str], limit: int = DEFAULT_DAILY_POINTERS) -> list[str]:
    """Most recent daily note paths first (names are date-stamped)."""
    return sorted(set(paths), reverse=True)[: max(int(limit), 0)]


def render_memory_index(
    entries: list[dict],
    daily_pointers: list[str],
    *,
    max_lines: int = INDEX_MAX_LINES,
    max_bytes: int = INDEX_MAX_BYTES,
) -> str:
    """Deterministic bounded render: line cap, then byte cap, then the notice.

    Same inputs always render byte-identical output; dropping entries from the
    tail keeps the head stable across rebuilds.
    """
    lines = list(_HEADER_LINES)
    if not entries:
        lines.append("- (no muscle SOPs mirrored yet)")
    for entry in entries:
        headline = entry.get("description") or "(no description)"
        lines.append(
            f"- [[{entry['path']}]] [{entry['verified_status']}] (scope={entry['scope']}) — {headline}",
        )
    lines.extend(_DAILY_HEADING)
    if not daily_pointers:
        lines.append("- (no recent daily notes)")
    for pointer in daily_pointers:
        lines.append(f"- [[{pointer}]]")
    lines.extend(_FOOTER_LINES)

    max_lines = max(int(max_lines), _MIN_RENDER_LINES)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1]
        lines.append(TRUNCATION_NOTICE)

    def rendered(current: list[str]) -> bytes:
        return ("\n".join(current) + "\n").encode("utf-8")

    while len(rendered(lines)) > max_bytes and len(lines) > _MIN_RENDER_LINES:
        # Drop the last entry line — the one before a trailing notice, if any —
        # so every iteration removes a non-notice line and makes progress.
        index = len(lines) - 2 if lines[-1] == TRUNCATION_NOTICE else len(lines) - 1
        if index <= 0:
            break
        lines.pop(index)
        if lines[-1] != TRUNCATION_NOTICE:
            lines.append(TRUNCATION_NOTICE)
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > max_bytes:  # pathological: cut the bytes hard
        text = text.encode("utf-8")[: max(int(max_bytes), 1)].decode("utf-8", errors="ignore")
    return text


@R.register("ops_reflex_index_step")
class OpsReflexIndexStep(BaseStep):
    """Rebuild ``metadata/ops_index/MEMORY.md`` from mirror + status ledger.

    Wired as the last step of the muscle-sync job (and its cron), so every
    sync — connector reconcile, cron, manual — leaves a fresh bounded index.
    Empty mirrors degrade to a header-only index instead of an error.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_rel = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        daily = str(self.config_value("daily_dir")).strip("/")
        daily_limit = max(int(self.kwargs.get("daily_limit", DEFAULT_DAILY_POINTERS)), 0)

        root = workspace / muscle_rel
        mirrors: dict[str, dict] = {}
        for file_path in sorted(root.rglob("*.md")) if root.is_dir() else []:
            if not file_path.is_file():
                continue
            try:
                post = frontmatter.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:  # noqa: BLE001 - unreadable mirrors are skipped, not fatal
                continue
            mirrors[file_path.relative_to(workspace).as_posix()] = {
                "description": str(post.metadata.get("description") or ""),
                "scope": str(post.metadata.get("scope") or "general"),
                "body": post.content,
            }
        ledger = load_status_ledger(audit_status_path(workspace, metadata_dir))
        entries = index_entries(mirrors, ledger)

        daily_root = workspace / daily
        daily_paths = (
            [p.relative_to(workspace).as_posix() for p in sorted(daily_root.rglob("*.md")) if p.is_file()]
            if daily_root.is_dir()
            else []
        )
        pointers = recent_daily_pointers(daily_paths, daily_limit)

        content = render_memory_index(entries, pointers)
        out_rel = f"{metadata_dir.strip('/')}/ops_index/MEMORY.md"
        out_path = workspace / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

        # Shared-Response contract: metadata only. The muscle-sync step owns
        # answer/success for this job; overwriting either would mask a cloud
        # outage recorded earlier in the same invocation.
        self.context.response.metadata["reflex_index"] = {
            "path": out_rel,
            "entries": len(entries),
            "daily_pointers": pointers,
            "lines": content.count("\n"),
            "bytes": len(content.encode("utf-8")),
            "truncated": TRUNCATION_NOTICE in content,
            "index_max_lines": INDEX_MAX_LINES,
            "index_max_bytes": INDEX_MAX_BYTES,
        }
        return self.context.response


# Re-exported for the vocabulary contract test; the index renders statuses
# from the same single definition the audit stamps and search weights read.
__all__ = [
    "DEFAULT_DAILY_POINTERS",
    "DEFAULT_VERIFIED_STATUS",
    "INDEX_MAX_BYTES",
    "INDEX_MAX_LINES",
    "OpsReflexIndexStep",
    "TRUNCATION_NOTICE",
    "index_entries",
    "recent_daily_pointers",
    "render_memory_index",
]
