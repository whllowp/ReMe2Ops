"""Cloud-side failure notes: write usage failures back into the SOP body.

Reference: JARVIS-1 (craftjarvis-jarvis1.github.io) cross-epoch self-improvement
— the missing-furnace failure of epoch 1 becomes the furnace step of epoch 2
and the extra-log fuel stock of epoch 3; every failure makes the next attempt
better prepared. Until now a failed outcome only lived in counters
(``usage.jsonl``, the audit candidate list) — the SOP body itself never
absorbed the lesson. This step closes the write-back loop: for each failed
usage record it appends a ``(date, query)`` line under a ``## Failure Notes``
section of the corresponding SOP (deduped per query, newest first, capped),
stamps ``failure_count`` into the frontmatter, and leaves everything else
untouched. A muscle canonical updated this way propagates through the existing
mtime sync → edge hash-mismatch → re-audit chain (dream_audit design intent).

Target resolution (review finding 4): the usage path names where the memory
was used, not necessarily where it lives now. Resolution order —

1. the path exists on the cloud disk → write it directly;
2. it does not exist (promoted: ``move`` relocated it, promote.py) → the
   muscle canonical is found via its ``promoted_from`` frontmatter alias and
   written there (the muscle copy is the single canonical; search already
   aliases it through the same field, tiered_search.py);
3. neither exists → skipped, recorded in ``metadata["failure_notes"]``.

This step only ever REWRITES existing files, never creates one — recreating
the old procedure path would feed the promotion scan again and its
``move(overwrite=True)`` would then clobber the muscle canonical (the
double-copy destruction chain).
"""

import re
from pathlib import Path

import frontmatter

from ...components import R
from ..base_step import BaseStep
from .dream_audit import load_json_state
from .usage import read_usage_records, usage_file_path

FAILURE_NOTES_HEADING = "## Failure Notes"
FAILURE_NOTES_CAP = 5
UNKNOWN_DATE = "unknown"
UNKNOWN_QUERY = "unrecorded query"

_FAILURE_SECTION = re.compile(r"^##[ \t]*Failure Notes[ \t]*$", re.M)
_HEADING = re.compile(r"^#{1,6}[ \t]", re.M)
_NOTE_LINE = re.compile(r"^- (.+)$")


def failure_index_path(workspace: Path, metadata_dir: str) -> Path:
    """Workspace path of the failure-notes cursor bookkeeping (reserved)."""
    return Path(workspace) / metadata_dir / "ops_failure_notes" / "state.json"


def load_failure_state(path: Path) -> dict:
    """Read the failure-notes state; tolerate a missing or corrupted file."""
    return load_json_state(path)


def note_date(ts: str) -> str:
    """``YYYY-MM-DD`` prefix of a usage ts, or ``unknown`` when unusable."""
    text = str(ts or "").strip()
    return text[:10] if len(text) >= 10 else UNKNOWN_DATE


def note_query(record: dict) -> str:
    """Query text of a failed record; session id, then a neutral placeholder."""
    query = str(record.get("query") or "").strip()
    if query:
        return query
    session = str(record.get("session_id") or "").strip()
    return f"session:{session}" if session else UNKNOWN_QUERY


def parse_failure_notes(body: str) -> list[tuple[str, str]]:
    """``[(date, query)]`` of an existing ## Failure Notes section (possibly empty)."""
    match = _FAILURE_SECTION.search(body or "")
    if match is None:
        return []
    start = match.end()
    nxt = _HEADING.search(body, start)
    end = nxt.start() if nxt else len(body)
    notes: list[tuple[str, str]] = []
    for line in body[start:end].splitlines():
        item = _NOTE_LINE.match(line.strip())
        if item is None:
            continue
        date, sep, query = item.group(1).partition(" — ")
        if sep:
            notes.append((date.strip(), query.strip()))
    return notes


def merge_failure_notes(
    existing: list[tuple[str, str]],
    incoming: list[tuple[str, str]],
    *,
    cap: int = FAILURE_NOTES_CAP,
) -> list[tuple[str, str]]:
    """Dedupe per query (newest date wins), sort newest-first, cap the oldest away.

    Deterministic: date descending, then query ascending within one date.
    """
    by_query: dict[str, str] = {}
    for date, query in [*existing, *incoming]:
        if query not in by_query or date > by_query[query]:
            by_query[query] = date
    notes = [(date, query) for query, date in by_query.items()]
    notes.sort(key=lambda item: item[1])  # secondary: query asc
    notes.sort(key=lambda item: item[0], reverse=True)  # primary: date desc (stable)
    return notes[: max(int(cap), 1)]


def render_failure_section(notes: list[tuple[str, str]]) -> str:
    """Deterministic section body for the merged notes."""
    lines = [FAILURE_NOTES_HEADING, ""]
    lines.extend(f"- {date} — {query}" for date, query in notes)
    return "\n".join(lines).strip("\n")


def apply_failure_notes(body: str, notes: list[tuple[str, str]]) -> str:
    """Body with the merged section replacing the old one (or appended at the end)."""
    if not notes:
        return body
    rendered = render_failure_section(notes)
    match = _FAILURE_SECTION.search(body or "")
    text = body or ""
    if match is not None:
        nxt = _HEADING.search(text, match.end())
        end = nxt.start() if nxt else len(text)
        return f"{text[:match.start()]}{rendered}\n\n{text[end:].lstrip('\n')}".rstrip("\n") + "\n"
    return f"{text.rstrip('\n')}\n\n{rendered}\n" if text.strip() else f"{rendered}\n"


def collect_failures(records: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Failed-only records grouped per usage path, deduped per (path, query)."""
    failures: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict) or str(record.get("outcome") or "").strip().lower() != "failed":
            continue
        path = str(record.get("path") or "").strip()
        if not path:
            continue
        date, query = note_date(record.get("ts")), note_query(record)
        seen = failures.setdefault(path, {})
        if query not in seen or date > seen[query]:
            seen[query] = date
    return {
        path: sorted(((date, query) for query, date in notes.items()), key=lambda item: item[1])
        for path, notes in failures.items()
    }


def promoted_canonicals(muscle_posts: dict[str, dict]) -> dict[str, str]:
    """``{promoted_from: muscle_path}`` from muscle frontmatter (sorted-first wins)."""
    canonicals: dict[str, str] = {}
    for muscle_path in sorted(muscle_posts):
        origin = str(muscle_posts[muscle_path].get("promoted_from") or "").strip()
        if origin and origin not in canonicals:
            canonicals[origin] = muscle_path
    return canonicals


@R.register("ops_failure_notes_step")
class OpsFailureNotesStep(BaseStep):
    """Write failed usage evidence back into the SOP body as failure notes.

    Resolves each failed usage path to an existing file (direct hit, or muscle
    canonical via ``promoted_from``), merges the ``(date, query)`` notes into
    its ``## Failure Notes`` section, and stamps ``failure_count`` in the
    frontmatter — rewriting existing files only, and only when the merged
    section actually changed (no spurious mtime bumps). Unsatisfiable paths
    land in ``metadata["failure_notes"]["skipped"]``; the step never creates a
    file, so the old procedure path can never be resurrected into a second
    copy. Job success accounting stays with the cron chain.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path.resolve()
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_dir = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        cap = max(int(self.kwargs.get("max_failure_notes", FAILURE_NOTES_CAP)), 1)

        records = read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True))
        failures = collect_failures(records)

        muscle_posts: dict[str, dict] = {}
        muscle_root = workspace / muscle_dir
        for file_path in sorted(muscle_root.rglob("*.md")) if muscle_root.is_dir() else []:
            try:
                muscle_posts[file_path.relative_to(workspace).as_posix()] = frontmatter.loads(
                    file_path.read_text(encoding="utf-8", errors="replace"),
                ).metadata
            except Exception:  # noqa: BLE001 - unreadable mirrors resolve to nothing
                continue
        canonicals = promoted_canonicals(muscle_posts)

        updated: list[dict] = []
        skipped: list[dict] = []
        for usage_path in sorted(failures):
            target = usage_path if (workspace / usage_path).is_file() else canonicals.get(usage_path, "")
            if not target or not (workspace / target).is_file():
                skipped.append({"path": usage_path, "reason": "no existing file or promoted canonical"})
                continue
            try:
                post = frontmatter.loads((workspace / target).read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001 - unreadable targets are skipped, never fatal
                skipped.append({"path": usage_path, "reason": f"unreadable target {target}: {type(exc).__name__}"})
                continue

            notes = merge_failure_notes(parse_failure_notes(post.content), failures[usage_path], cap=cap)
            new_body = apply_failure_notes(post.content, notes)
            # Trailing-newline-only differences are not a change (the markdown
            # writer normalizes the final newline; frontmatter parsing may
            # strip it) — skipping keeps the mtime stable for the sync loop.
            if new_body.rstrip("\n") == post.content.rstrip("\n"):
                continue  # already recorded; no write, no mtime bump

            metadata = {key: value for key, value in post.metadata.items() if key not in ("name", "description")}
            metadata["failure_count"] = len(notes)
            try:
                response = await self.run_job(
                    "write",
                    path=target,
                    name=str(post.metadata.get("name") or Path(target).stem),
                    description=str(post.metadata.get("description") or f"SOP with failure notes ({len(notes)})"),
                    content=new_body,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001 - one failed write skips, never fatal
                skipped.append({"path": usage_path, "reason": f"write failed on {target}: {type(exc).__name__}: {exc}"})
                continue
            if not response.success:
                skipped.append({"path": usage_path, "reason": f"write failed on {target}: {response.answer}"})
                continue
            updated.append({"usage_path": usage_path, "path": target, "failure_count": len(notes), "notes": [f"{date} — {query}" for date, query in notes]})

        self.context.response.metadata["failure_notes"] = {
            "failures_seen": sum(len(notes) for notes in failures.values()),
            "updated": updated,
            "skipped": skipped,
            "canonicals": len(canonicals),
        }
        return self.context.response
