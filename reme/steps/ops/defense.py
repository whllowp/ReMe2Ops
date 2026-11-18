"""Pre-ingest defense scan: redact or block PII/secret hits before storage.

Reference: Hindsight's ``Memory Defense`` (vectorize-io/hindsight) — per-store
opt-in PII/secret scanning with a pattern pack that scrubs or refuses content
before it is persisted. This module ports the idea onto the ops dream channel:
every memory unit's ``name``/``summary`` is scanned with a pure ``re`` pattern
pack (no LLM) at two gates —

- **edge outbound** (``ops_dream_push`` anchor chain, between
  ``dream_extract_step`` and ``ops_push_step``): ``redact`` mode rewrites each
  hit to a ``[REDACTED:<pattern>]`` placeholder and the unit continues
  downstream;
- **cloud inbound** (``cloud_ingest`` chain, after ``ops_ingest_seed_step``):
  ``block`` mode drops the unit once, explicitly.

Block semantics (review finding 5): a blocked unit's source paths stay in
``state.processed_paths`` and never enter ``state.failed_paths``, so
``dream_finish_step`` consumes the checkpoint (it only requires
``p in processed_paths and p not in failed_paths``) and the next run does NOT
re-extract them — a one-shot explicit discard. Re-extraction would only
produce the same secret again; routing the paths through ``failed_paths``
would create an eternal re-block loop. Blocked reasons ride the existing
``metadata["rejected"]`` channel (``ops_ingest_seed_step`` already returns
rejection reasons to the edge), and a daily defense report records the
blocked paths and pattern types so a discard is auditable, never silent.
Matched secret values are never written anywhere.
"""

import datetime as dt
import re
from typing import NamedTuple

from ...components import R
from ..base_step import BaseStep
from ..evolve.dream.utils import state_from_context, store_state, today

DEFENSE_REDACT = "redact"
DEFENSE_BLOCK = "block"
DEFENSE_MODES = (DEFENSE_REDACT, DEFENSE_BLOCK)
DEFAULT_DEFENSE_MODE = DEFENSE_REDACT

# Placeholder written over a matched secret value; contains no characters any
# pattern below would match, so substitutions can never cascade.
REDACT_TEMPLATE = "[REDACTED:{name}]"


class DefensePattern(NamedTuple):
    """One named regex; the sensitive span is always the ``secret`` group."""

    name: str
    regex: "re.Pattern[str]"


def _compile(name: str, source: str, flags: int = 0) -> DefensePattern:
    return DefensePattern(name=name, regex=re.compile(source, flags))


# Format-specific keys first (so ``api_key = "sk-..."`` reports the concrete
# format), then generic assignment/credential shapes, then network internals.
# Every pattern carries a ``secret`` group spanning exactly the text to erase.
DEFENSE_PATTERNS: tuple[DefensePattern, ...] = (
    _compile("private_key_block", r"(?P<secret>-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----)", re.S),
    _compile("openai_style_key", r"(?P<secret>\bsk-(?:proj-)?[A-Za-z0-9_\-]{16,})"),
    _compile("github_token", r"(?P<secret>\bgh[pousr]_[A-Za-z0-9]{16,}\b)"),
    _compile("aws_access_key_id", r"(?P<secret>\bAKIA[0-9A-Z]{16}\b)"),
    _compile("jwt_token", r"(?P<secret>\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,})"),
    _compile("slack_token", r"(?P<secret>\bxox[baprs]-[A-Za-z0-9\-]{8,})"),
    _compile("bearer_token", r"(?P<secret>\bBearer\s+[A-Za-z0-9._\-/=+]{16,})", re.I),
    _compile(
        "api_key_assignment",
        r"(?i)\bapi[_-]?key\b[ \t]*[=:][ \t]*(?P<quote>[\"']?)(?P<secret>[^\s\"']{8,})(?P=quote)",
    ),
    _compile(
        "credential_assignment",
        r"(?i)\b(?:password|passwd|pwd|secret|access[_-]?token|auth[_-]?token|client[_-]?secret|token)\b"
        r"[ \t]*[=:][ \t]*(?P<quote>[\"']?)(?P<secret>[^\s\"']{6,})(?P=quote)",
    ),
    _compile("url_credentials", r"(?P<secret>(?<=://)[^\s/:@?#]+:[^\s/@?#]+@)"),
    _compile(
        "internal_ipv4",
        r"(?P<secret>\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b)",
    ),
)


def _substitute(match: "re.Match[str]", name: str) -> str:
    """Replace only the ``secret`` span of one match with the placeholder."""
    secret = match.group("secret")
    if not secret:
        return match.group(0)
    return match.group(0).replace(secret, REDACT_TEMPLATE.format(name=name), 1)


def scan_text(text: str) -> list[str]:
    """Sorted names of the defense patterns hit by one text field."""
    if not text:
        return []
    return sorted({pattern.name for pattern in DEFENSE_PATTERNS if pattern.regex.search(text)})


def redact_text(text: str) -> tuple[str, list[str]]:
    """Rewrite every hit to a ``[REDACTED:<pattern>]`` placeholder.

    Patterns are applied in :data:`DEFENSE_PATTERNS` order; a placeholder can
    never match a later pattern, so applications are independent. Returns
    ``(text, applied_pattern_names)``.
    """
    if not text:
        return text, []
    out = text
    applied: list[str] = []
    for pattern in DEFENSE_PATTERNS:
        replaced = pattern.regex.sub(lambda m, _name=pattern.name: _substitute(m, _name), out)
        if replaced != out:
            applied.append(pattern.name)
            out = replaced
    return out, applied


def scan_unit(unit: dict) -> list[str]:
    """Sorted union of pattern names hit across one unit's name and summary."""
    hits = set(scan_text(str(unit.get("name") or "")))
    hits.update(scan_text(str(unit.get("summary") or "")))
    return sorted(hits)


def redact_unit(unit: dict) -> tuple[dict, list[str]]:
    """Copy of ``unit`` with name/summary rewritten; other fields untouched."""
    name, name_hits = redact_text(str(unit.get("name") or ""))
    summary, summary_hits = redact_text(str(unit.get("summary") or ""))
    scrubbed = dict(unit)
    scrubbed["name"] = name
    scrubbed["summary"] = summary
    return scrubbed, sorted(set(name_hits) | set(summary_hits))


def build_defense_note(mode: str, blocked: list[dict], redacted: list[dict]) -> str:
    """Markdown audit report: blocked/redacted paths plus pattern types only.

    Matched secret values are deliberately never recorded — the report must be
    safe to keep in the workspace it defends.
    """
    lines = [
        "# Ops defense scan report",
        "",
        f"Mode: {mode}. Blocked units were discarded once and are not retried;",
        "matched values are never recorded here — only paths and pattern types.",
        "",
    ]
    if blocked:
        lines.append("## Blocked")
        for entry in blocked:
            lines.append(f"- {entry['name']} — patterns: {', '.join(entry['patterns'])}")
            lines.extend(f"  - [[{path}]]" for path in entry["paths"])
    else:
        lines.extend(["## Blocked", "", "- (none)"])
    lines.append("")
    if redacted:
        lines.append("## Redacted")
        lines.extend(f"- {entry['name']} — patterns: {', '.join(entry['patterns'])}" for entry in redacted)
    else:
        lines.extend(["## Redacted", "", "- (none)"])
    return "\n".join(lines).strip() + "\n"


@R.register("ops_defense_scan_step")
class OpsDefenseScanStep(BaseStep):
    """Scan dream units for PII/secret patterns before they are stored or pushed.

    Runs mid-chain on the shared dream state (``DreamState.units``), so it
    writes only ``metadata["defense"]`` and leaves ``answer``/``success`` to
    the chain's terminal steps. In ``redact`` mode hits are rewritten in place
    and units continue downstream; in ``block`` mode hitting units are removed
    from ``state.units`` while their source paths stay checkpointable
    (``processed_paths``, never ``failed_paths``), the shared
    ``metadata["rejected"]`` channel records why, and a daily defense report
    lands under ``daily/<date>/ops-defense-<stamp>.md``. Units without hits
    pass through untouched (the original dict object is kept, not copied).
    """

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        mode = str(self.kwargs.get("mode") or DEFAULT_DEFENSE_MODE).strip().lower()
        if mode not in DEFENSE_MODES:
            mode = DEFAULT_DEFENSE_MODE

        redacted: list[dict] = []
        blocked: list[dict] = []
        kept: list[dict] = []
        for unit in state.units:
            hits = scan_unit(unit) if isinstance(unit, dict) else []
            if not hits:
                kept.append(unit)  # clean units pass through byte-identical
                continue
            paths = [str(path) for path in (unit.get("paths") or [])]
            if mode == DEFENSE_BLOCK:
                blocked.append({"name": str(unit.get("name") or ""), "patterns": hits, "paths": paths})
                # One-shot explicit discard: keep the source paths checkpointable
                # and out of failed_paths so dream_finish_step consumes the
                # checkpoint instead of retrying forever.
                if state.processed_paths is not None:
                    for path in paths:
                        if path not in state.processed_paths:
                            state.processed_paths.append(path)
                continue
            scrubbed, applied = redact_unit(unit)
            kept.append(scrubbed)
            redacted.append({"name": str(scrubbed.get("name") or ""), "patterns": applied, "paths": paths})
        state.units = kept
        store_state(self, state)

        blocked_paths = sorted({path for entry in blocked for path in entry["paths"]})
        if blocked:
            rejected = self.context.response.metadata.get("rejected")
            if not isinstance(rejected, list):
                rejected = []
            rejected.extend(f"defense blocked unit {entry['name']!r}: {'+'.join(entry['patterns'])}" for entry in blocked)
            self.context.response.metadata["rejected"] = rejected

        note_path = ""
        report_error = ""
        if blocked:
            day = state.date or today(self, str(self.context.get("date") or ""))
            daily = str(state.daily_dir or self.config_value("daily_dir")).strip("/")
            stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            note_path = f"{daily}/{day}/ops-defense-{stamp}.md"
            try:
                await self.run_job(
                    "write",
                    path=note_path,
                    name=f"ops-defense-{stamp}",
                    description=f"Defense scan blocked {len(blocked)} unit(s)",
                    content=build_defense_note(mode, blocked, redacted),
                    metadata={
                        "source": "ops_defense_scan",
                        "mode": mode,
                        "blocked_paths": blocked_paths,
                        "redacted_units": [entry["name"] for entry in redacted],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - a failed report must not break ingest
                report_error = f"{type(exc).__name__}: {exc}"
                note_path = ""

        self.context.response.metadata["defense"] = {
            "mode": mode,
            "units_in": len(kept) + len(blocked),
            "units_out": len(kept),
            "redacted": redacted,
            "blocked": blocked,
            "blocked_paths": blocked_paths,
            "note": note_path,
            "report_error": report_error,
        }
        return self.context.response
