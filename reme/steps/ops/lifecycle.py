"""Muscle lifecycle: heat decay, demotion tombstones, reversible demotion.

The memery architecture proposal's lifecycle (write -> active -> demoted ->
retired) with the reversibility ladder: cold muscle SOPs are *demoted* back to
the long-term procedure bucket — never deleted — carrying tombstone frontmatter
(``valid_until`` / ``superseded_by`` / ``lifecycle``). Demotion is reversible:
a later promotion simply moves the file back (the byte-level revival rung).

Heat follows the proposal's decay formula: exponential recency decay over a
half-life plus importance (success rate) and access (hit count), so an
outdated SOP that keeps failing cools below the demotion bar while a hot one
stays. Every demotion broadcasts ``muscle.updated`` so attached edges drop the
stale mirror immediately — negative updates are first-class.
"""

import datetime as dt
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .connector import broadcast_muscle_update
from .usage import aggregate_usage, now_iso, read_usage_records, usage_file_path

HALF_LIFE_DAYS = 14.0
COLD_THRESHOLD = 0.25


def parse_ts(raw: str):
    """Best-effort ISO timestamp parse; None when unusable."""
    try:
        return dt.datetime.fromisoformat(str(raw).strip())
    except (TypeError, ValueError):
        return None


def heat_score(*, last_used: str, success_rate: float, hits: int, hit_cap: float = 10.0,
               half_life_days: float = HALF_LIFE_DAYS, now: dt.datetime | None = None) -> float:
    """Weighted heat in [0, 1]: recency decay + importance + effective access.

    memery formula shape with one correction: access volume only counts when
    it succeeded (``access * success_rate``) - a stale, all-failing SOP cannot
    keep itself hot on raw hit volume. Weights (0.4, 0.3, 0.3).
    """
    now = now or dt.datetime.now()
    last = parse_ts(last_used)
    # An unparseable last-use earns no recency credit: unknown age must not
    # keep a possibly-ancient SOP hot (conservative for demotion).
    recency = 0.0
    if last is not None:
        age_days = max((now - last).total_seconds() / 86400.0, 0.0)
        recency = 2.0 ** (-age_days / max(half_life_days, 1e-6))
    rate = min(max(float(success_rate or 0.0), 0.0), 1.0)
    access = min(float(hits or 0) / max(float(hit_cap), 1.0), 1.0) * rate
    return round(0.4 * recency + 0.3 * rate + 0.3 * access, 4)


def is_cold(heat: float, threshold: float = COLD_THRESHOLD) -> bool:
    return heat < threshold


@R.register("ops_decay_scan_step")
class OpsDecayScanStep(BaseStep):
    """Cool-down sweep: demote muscle SOPs whose heat fell below the bar.

    Promotion scan keeps muscle honest on the way in; this scan keeps it honest
    on the way out. Demotion moves the file back to the procedure bucket with a
    tombstone stamp (memery: demotion-rung reversibility, no deletion), and
    broadcasts the removal to every connected client.
    """

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_dir = str(self.kwargs.get("muscle_dir") or f"{digest_dir}/muscle").strip("/")
        procedure_dir = str(self.kwargs.get("procedure_dir") or f"{digest_dir}/procedure").strip("/")
        threshold = float(self.kwargs.get("cold_threshold", COLD_THRESHOLD))
        half_life_days = float(self.kwargs.get("half_life_days", HALF_LIFE_DAYS))
        hit_cap = float(self.kwargs.get("hit_cap", 10.0))
        max_demotions = max(int(self.kwargs.get("max_demotions_per_run", 3)), 1)

        usage_stats = aggregate_usage(read_usage_records(usage_file_path(workspace, metadata_dir, cloud=True)))
        muscle_root = workspace / muscle_dir
        audited, demoted, kept_hot = [], [], []
        if muscle_root.is_dir():
            for file_path in sorted(muscle_root.rglob("*.md")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(workspace).as_posix()
                stats = usage_stats.get(rel) or {}
                heat = heat_score(
                    last_used=str(stats.get("last_used") or ""),
                    success_rate=float(stats.get("success_rate") or 0.0),
                    hits=int(stats.get("total") or 0),
                    hit_cap=hit_cap,
                    half_life_days=half_life_days,
                )
                entry = {"path": rel, "heat": heat}
                audited.append(entry)
                if not is_cold(heat, threshold):
                    kept_hot.append(entry)
                    continue
                if len(demoted) >= max_demotions:
                    continue
                dst = f"{procedure_dir}/{Path(rel).name}"
                move_response = await self.run_job("move", src_path=rel, dst_path=dst, overwrite=True)
                if not move_response.success:
                    entry["error"] = str(move_response.answer)
                    demoted.append(entry)
                    continue
                tombstone = {
                    "lifecycle": "demoted",
                    "memory_type": "procedure",
                    "valid_until": now_iso(),
                    "demoted_at": now_iso(),
                    "demote_reason": f"heat {heat} below {threshold}",
                    "heat": heat,
                }
                await self.run_job("frontmatter_update", path=dst, metadata=tombstone)
                demoted.append({**entry, "demoted_to": dst})
                await broadcast_muscle_update(self, [rel], reason="demoted")

        self.context.response.answer = (
            f"Decay scan: {len(audited)} muscle SOP(s) audited, {len(demoted)} demoted, "
            f"{len(kept_hot)} still hot (threshold={threshold}, half_life={half_life_days}d)"
        )
        self.context.response.metadata["decay"] = {
            "audited": audited,
            "demoted": demoted,
            "kept_hot": kept_hot,
            "threshold": threshold,
        }
        return self.context.response
