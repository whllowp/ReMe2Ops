"""Usage-record storage and aggregation shared by ops feedback and promotion.

A usage record is one JSONL line: ``{ts, path, outcome, query?, session_id?}``
where ``path`` is a workspace-relative digest file and ``outcome`` is ``solved``
or ``failed``. The edge records outcomes per query hit and syncs deltas to the
cloud, whose aggregate drives muscle promotion.
"""

import datetime as dt
import json
from pathlib import Path

VALID_OUTCOMES = ("solved", "failed")


def usage_file_path(workspace: Path, metadata_dir: str, *, cloud: bool = False) -> Path:
    """Return the workspace path of the usage JSONL (edge feedback or cloud aggregate)."""
    sub = "ops_usage" if cloud else "ops_feedback"
    return workspace / metadata_dir / sub / "usage.jsonl"


def read_usage_records(path: Path) -> list[dict]:
    """Read usage records; tolerate a missing or partially corrupted file."""
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def append_usage_record(path: Path, record: dict) -> None:
    """Append one usage record as a JSONL line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_outcome(raw) -> str | None:
    """Return the canonical outcome string, or None when invalid."""
    value = str(raw or "").strip().lower()
    return value if value in VALID_OUTCOMES else None


def now_iso() -> str:
    """Current local timestamp for usage and promotion records."""
    return dt.datetime.now().isoformat(timespec="seconds")


def aggregate_usage(records: list[dict]) -> dict[str, dict]:
    """Aggregate records by digest path into counters.

    Returns ``{path: {solved, failed, total, success_rate, last_used, devices,
    device_list}}``; records with an unknown path or outcome are ignored. The
    ``devices``/``device_list`` fields back the multi-objective breadth term.
    """
    stats: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        path = str(record.get("path") or "").strip()
        outcome = normalize_outcome(record.get("outcome"))
        if not path or outcome is None:
            continue
        entry = stats.setdefault(
            path,
            {"solved": 0, "failed": 0, "total": 0, "success_rate": 0.0, "last_used": "", "devices": 0, "device_list": []},
        )
        entry[outcome] += 1
        entry["total"] += 1
        entry["success_rate"] = round(entry["solved"] / entry["total"], 4)
        last_used = str(record.get("ts") or "")
        if last_used > entry["last_used"]:
            entry["last_used"] = last_used
        device = str(record.get("source_device") or "").strip()
        if device and device not in entry["device_list"]:
            entry["device_list"].append(device)
            entry["devices"] = len(entry["device_list"])
    return stats
