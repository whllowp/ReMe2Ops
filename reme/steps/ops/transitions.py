"""Time-aware SOP transition graphs (SHARP §2, scaled to ops memory).

Global aggregation hides temporal regularities (the paper's Coffee→Office
31% global vs 58% weekday-morning). The same holds for ops: "disk alert →
cleanup runbook" dominates nights, "cert alert → renewal runbook" dominates
weekday mornings. Two granularities are kept: one global view plus time-
conditioned views (6 hour bins × week type), built incrementally from the
usage aggregate. Retrieval consults the views matching the *current* time and
returns structural support scores that the tiered search can use as a small
ranking bonus.
"""

import datetime as dt
import json
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .usage import read_usage_records, usage_file_path

HOUR_BINS = (
    ("late_night", 0, 6),
    ("morning", 6, 11),
    ("noon", 11, 14),
    ("afternoon", 14, 18),
    ("evening", 18, 22),
    ("night", 22, 24),
)

GLOBAL_VIEW = "global"


def hour_bin_name(hour: int) -> str:
    for name, start, end in HOUR_BINS:
        if start <= hour < end:
            return name
    return HOUR_BINS[0][0]


def week_type_name(weekdays: int) -> str:
    """weekday (Mon-Fri) or weekend (Sat/Sun)."""
    return "weekday" if weekdays < 5 else "weekend"


def classify_time(raw: str) -> dict:
    """Decompose a timestamp into the view keys it belongs to."""
    moment = parse_ts(raw)
    if moment is None:
        return {}
    wt = week_type_name(moment.weekday())
    hb = hour_bin_name(moment.hour)
    return {"global": GLOBAL_VIEW, "week_type": f"wt:{wt}", "hour_bin": f"bin:{hb}",
            "joint": f"wt:{wt}+bin:{hb}"}


def parse_ts(raw):
    try:
        return dt.datetime.fromisoformat(str(raw).strip())
    except (TypeError, ValueError):
        return None


def category_of(path: str) -> str:
    """SOP category = filename stem (frontmatter categories later refine this)."""
    from pathlib import Path as _P

    return _P(path).stem if path else ""


def build_transition_graphs(records: list[dict]) -> dict:
    """Incremental transition counts per view.

    Consecutive usage events of one device (timestamp order) form the
    transitions ``category(prev) -> category(curr)``; each transition is added
    to the global view and every conditional view selected by the *destination*
    event's time (SHARP's construction rule).
    """
    by_device: dict[str, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        device = str(record.get("source_device") or "").strip() or "unknown"
        by_device.setdefault(device, []).append(record)
    views: dict[str, dict[str, float]] = {}
    for device, events in by_device.items():
        events.sort(key=lambda r: str(r.get("ts") or ""))
        previous = None
        for record in events:
            current = category_of(str(record.get("path") or ""))
            if not current:
                previous = None
                continue
            if previous and previous != current:
                keys = classify_time(str(record.get("ts") or ""))
                for key in keys.values():
                    edges = views.setdefault(key, {})
                    edge = f"{previous}->{current}"
                    edges[edge] = edges.get(edge, 0.0) + 1.0
            previous = current
    return views


def normalize_view(edges: dict[str, float]) -> dict[str, float]:
    """Turn counts into an outgoing distribution per source category."""
    out: dict[str, dict[str, float]] = {}
    for edge, weight in edges.items():
        source, _, target = edge.partition("->")
        bucket = out.setdefault(source, {})
        bucket[target] = bucket.get(target, 0.0) + weight
    total = {source: sum(targets.values()) for source, targets in out.items()}
    return {source: {t: round(w / total[source], 4) for t, w in targets.items()} for source, targets in out.items()}


def transition_support(graphs: dict, from_category: str, at_time: str, target_category: str) -> float:
    """Weighted support for `from -> target` at `at_time` (conditioned views 2x global)."""
    keys = classify_time(at_time)
    support = 0.0
    weight_total = 0.0
    for key, weight in ((keys.get("global"), 1.0), (keys.get("week_type"), 1.5),
                        (keys.get("hour_bin"), 1.5), (keys.get("joint"), 2.0)):
        if not key:
            continue
        edges = graphs.get(key) or {}
        probability = (normalize_view(edges).get(from_category) or {}).get(target_category)
        if probability is not None:
            support += weight * probability
            weight_total += weight
    return round(support / weight_total, 4) if weight_total else 0.0


def transitions_path(workspace, metadata_dir: str) -> Path:
    return Path(workspace) / metadata_dir / "ops_transitions" / "graphs.json"


def save_graphs(workspace, metadata_dir: str, graphs: dict) -> Path:
    path = transitions_path(workspace, metadata_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"built_at": now_iso(), "views": graphs}, ensure_ascii=False), encoding="utf-8")
    return path


def load_graphs(workspace, metadata_dir: str) -> dict:
    path = transitions_path(workspace, metadata_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("views") if isinstance(data, dict) else {}


def now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now().isoformat(timespec="seconds")


@R.register("ops_transitions_step")
class OpsTransitionsStep(BaseStep):
    """Rebuild the time-aware transition graphs from the usage aggregate."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        records = read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True))
        graphs = build_transition_graphs(records)
        path = save_graphs(workspace, metadata_dir, graphs)
        self.context.response.answer = (
            f"Transition graphs rebuilt: {len(graphs)} view(s) from {len(records)} record(s)"
        )
        self.context.response.metadata["transitions"] = {
            "views": sorted(graphs),
            "path": path.relative_to(workspace).as_posix(),
        }
        return self.context.response
