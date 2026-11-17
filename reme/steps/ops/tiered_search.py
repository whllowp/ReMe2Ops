"""Tiered ops memory search: muscle reflex -> short-term daily -> cloud long-term.

The edge answers from what it holds locally first (muscle mirror = conditioned
reflex, daily notes = working memory) and falls back to the cloud long-term
digest per query. Tier weights express the muscle-memory priority; usage
outcomes (solved/failed feedback) boost repeatedly-successful SOPs so the top
hit is the best known solution, not merely a lexically close one. The learning
loop adds a muscle-layer-only verified-status weighting: a hash-gated
``confirmed`` ledger entry boosts, a ``conflict`` penalizes — daily is
transient working memory and cloud is remote, so neither carries a status.
"""

import asyncio
from pathlib import Path

from ...components import R
from ..base_step import BaseStep
from .cloud import CloudClient, cloud_settings
from .dream_audit import (
    STATUS_CONFIRMED,
    STATUS_CONFLICT,
    VERIFIED_STATUS_VALUES,
    audit_status_path,
    ledger_status_for,
    load_status_ledger,
    read_body_text,
)
from .transitions import category_of, load_graphs, transition_support
from .usage import aggregate_usage, read_usage_records, usage_file_path

TIER_MUSCLE = "muscle"
TIER_DAILY = "daily"
TIER_CLOUD = "cloud"

_SNIPPET_CHARS = 200


def best_path_scores(results: list[dict]) -> dict[str, dict]:
    """Collapse chunk-level search results to one best entry per file path."""
    best: dict[str, dict] = {}
    for chunk in results:
        if not isinstance(chunk, dict):
            continue
        path = str(chunk.get("path") or "")
        if not path:
            continue
        score = float((chunk.get("scores") or {}).get("score") or 0.0)
        current = best.get(path)
        if current is None or score > current["score"]:
            best[path] = {"path": path, "score": score, "snippet": str(chunk.get("text") or "")[:_SNIPPET_CHARS]}
    return best


def merge_tier_hits(
    tiers: list[dict],
    *,
    limit: int,
    usage_stats: dict[str, dict],
    usage_boost: float,
    usage_aliases: dict[str, str] | None = None,
    status_by_path: dict[str, str] | None = None,
    confirmed_boost: float = 0.0,
    conflict_penalty: float = 0.0,
) -> list[dict]:
    """Merge per-tier ranked hit lists into one deduplicated, usage-boosted ranking.

    Each tier contributes ``weight / (rank + 1)`` per hit (rank is 1-based
    within that tier); every recorded ``solved`` outcome for the path adds
    ``usage_boost``. ``usage_aliases`` maps promoted muscle paths to their
    pre-promotion digest paths, so usage history accumulated before promotion
    keeps boosting the file afterwards. Ties therefore favour the reflex tier,
    then recency of success.

    ``status_by_path`` carries hash-gated verified statuses (muscle layer only,
    from the audit ledger): ``confirmed`` adds ``confirmed_boost`` and
    ``conflict`` subtracts ``conflict_penalty``; anything else — including a
    hash-mismatched entry the caller already dropped — stays neutral.
    """
    merged: dict[str, dict] = {}
    for tier in tiers:
        name = str(tier.get("tier") or "")
        weight = float(tier.get("weight") or 0.0)
        for rank, hit in enumerate(tier.get("hits") or [], start=1):
            path = str(hit.get("path") or "")
            if not path:
                continue
            item = merged.setdefault(path, {"path": path, "tiers": [], "score": 0.0})
            if name not in item["tiers"]:
                item["tiers"].append(name)
            item["score"] += weight / (rank + 1)
            for key in ("name", "description", "snippet"):
                if hit.get(key) and key not in item:
                    item[key] = hit[key]
    usage_aliases = usage_aliases or {}
    status_by_path = status_by_path or {}
    for path, item in merged.items():
        usage = combined_usage(path, usage_stats, usage_aliases)
        item["usage"] = usage
        if usage["solved"] and usage_boost > 0.0:
            item["score"] += usage_boost * usage["solved"]
            item["usage_boosted"] = True
        status = status_by_path.get(path)
        if status:
            item["verified_status"] = status
            if status == STATUS_CONFIRMED and confirmed_boost > 0.0:
                item["score"] += confirmed_boost
            elif status == STATUS_CONFLICT and conflict_penalty > 0.0:
                item["score"] -= conflict_penalty
    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return ranked[: max(int(limit), 1)]


def combined_usage(path: str, usage_stats: dict[str, dict], usage_aliases: dict[str, str]) -> dict:
    """Sum usage counters for a path and its pre-promotion alias, if any."""
    solved = failed = 0
    for lookup in (path, usage_aliases.get(path)):
        stats = usage_stats.get(lookup) if lookup else None
        if stats:
            solved += int(stats.get("solved") or 0)
            failed += int(stats.get("failed") or 0)
    return {"solved": solved, "failed": failed}


@R.register("ops_search_step")
class OpsSearchStep(BaseStep):
    """Find the best ops solution across muscle, short-term, and cloud tiers."""

    async def execute(self):
        assert self.context is not None
        raw_query = self.context.get("query")
        query = "" if raw_query is None else str(raw_query).strip()
        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        limit = max(int(self.context.get("limit") or self.kwargs.get("limit") or 5), 1)
        tier_limit = max(int(self.kwargs.get("tier_limit", 5)), 1)
        usage_boost = float(self.kwargs.get("usage_boost", 0.05))
        confirmed_boost = float(self.kwargs.get("confirmed_boost", 0.0))
        conflict_penalty = float(self.kwargs.get("conflict_penalty", 0.0))
        digest_dir = str(self.config_value("digest_dir")).strip("/")
        muscle_prefix = str(self.kwargs.get("muscle_prefix") or f"{digest_dir}/muscle").strip("/")
        daily_prefix = str(self.kwargs.get("daily_prefix") or self.config_value("daily_dir")).strip("/")
        weights = {
            TIER_MUSCLE: float(self.kwargs.get("muscle_weight", 1.0)),
            TIER_DAILY: float(self.kwargs.get("daily_weight", 0.6)),
            TIER_CLOUD: float(self.kwargs.get("cloud_weight", 0.4)),
        }

        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        usage_stats = aggregate_usage(read_usage_records(usage_file_path(workspace, metadata_dir)))
        transition_boost = float(self.kwargs.get("transition_boost", 0.0))
        graphs = load_graphs(workspace, metadata_dir) if transition_boost > 0.0 else {}
        last_category = str(self.context.get("last_category") or self.kwargs.get("last_category") or "").strip()

        async def local_tier(tier: str, prefix: str) -> dict:
            if not prefix:
                return {"tier": tier, "hits": [], "weight": weights[tier], "error": "empty prefix"}
            try:
                response = await self.run_job(
                    "search",
                    query=query,
                    limit=tier_limit,
                    search_filter={"path_prefixes": [prefix]},
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"[{self.name}] {tier} tier failed: {type(exc).__name__}: {exc}")
                return {"tier": tier, "hits": [], "weight": weights[tier], "error": str(exc)}
            if not response.success:
                return {"tier": tier, "hits": [], "weight": weights[tier], "error": str(response.answer)}
            ranked = sorted(
                best_path_scores(response.metadata.get("results") or []).values(),
                key=lambda hit: hit["score"],
                reverse=True,
            )
            return {"tier": tier, "hits": ranked, "weight": weights[tier], "error": None}

        async def cloud_tier() -> dict:
            host, port, timeout = cloud_settings(self)
            try:
                async with CloudClient(host, port, timeout) as cloud:
                    data = await cloud.post("node_search", {"query": query, "limit": tier_limit})
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"[{self.name}] cloud tier unreachable at {host}:{port}: {exc}")
                return {"tier": TIER_CLOUD, "hits": [], "weight": weights[TIER_CLOUD], "error": str(exc)}
            if not data.get("success", False):
                error = str(data.get("answer") or "cloud node_search failed")
                return {"tier": TIER_CLOUD, "hits": [], "weight": weights[TIER_CLOUD], "error": error}
            hits = [
                {
                    "path": str(hit.get("path") or ""),
                    "name": hit.get("name"),
                    "description": hit.get("description"),
                    "score": float(hit.get("score") or 0.0),
                }
                for hit in (data.get("metadata") or {}).get("hits") or []
                if hit.get("path")
            ]
            return {"tier": TIER_CLOUD, "hits": hits, "weight": weights[TIER_CLOUD], "error": None}

        muscle_tier, daily_tier, cloud_result = await asyncio.gather(
            local_tier(TIER_MUSCLE, muscle_prefix),
            local_tier(TIER_DAILY, daily_prefix),
            cloud_tier(),
        )
        tiers = [muscle_tier, daily_tier, cloud_result]
        usage_aliases, frontmatter_statuses = await self._frontmatter_flags(muscle_tier)
        status_by_path = self._ledger_statuses(muscle_tier, workspace, metadata_dir)
        results = merge_tier_hits(
            tiers,
            limit=limit,
            usage_stats=usage_stats,
            usage_boost=usage_boost,
            usage_aliases=usage_aliases,
            status_by_path=status_by_path,
            confirmed_boost=confirmed_boost,
            conflict_penalty=conflict_penalty,
        )
        if transition_boost > 0.0 and graphs and last_category:
            # SHARP-style time-conditioned structural support as a light bonus.
            import datetime as _dt

            now_text = _dt.datetime.now().isoformat(timespec="seconds")
            for item in results:
                support = transition_support(graphs, last_category, now_text, category_of(item["path"]))
                if support > 0.0:
                    item["score"] = round(item["score"] + transition_boost * support, 4)
                    item["transition_support"] = support
            results.sort(key=lambda item: item["score"], reverse=True)
            results = results[:limit]

        answer_lines = []
        for item in results:
            tier_label = "+".join(item["tiers"])
            usage_label = f" solved×{item['usage']['solved']}" if item["usage"]["solved"] else ""
            answer_lines.append(f"[{item['score']:.4f}] {item['path']} ({tier_label}){usage_label}")
            headline = item.get("name") or item.get("description") or item.get("snippet") or ""
            if headline:
                answer_lines.append(f"  {headline}")
        self.context.response.answer = "\n".join(answer_lines) or "No matching memory in any tier"
        self.context.response.metadata["results"] = results
        self.context.response.metadata["tiers"] = {
            tier["tier"]: {"hits": len(tier.get("hits") or []), "error": tier.get("error")} for tier in tiers
        }
        self.context.response.metadata["usage_boost"] = usage_boost
        if frontmatter_statuses:
            # Decorative stamps read in the same single get_nodes pass as the
            # promotion aliases; informational only, weighting uses the ledger.
            self.context.response.metadata["frontmatter_status"] = frontmatter_statuses
        return self.context.response

    def _ledger_statuses(self, muscle_tier: dict, workspace, metadata_dir: str) -> dict[str, str]:
        """Hash-gated verified statuses for muscle hits, from the audit ledger.

        This is the only weighting source: a ledger entry whose recorded
        ``body_sha256`` no longer matches the mirror body is dropped, so a
        mirror rewritten by muscle_sync falls back to neutral instead of
        weighting a stale verdict. Daily/cloud hits never appear here.
        """
        hit_paths = sorted({hit["path"] for hit in muscle_tier.get("hits") or []})
        if not hit_paths:
            return {}
        ledger = load_status_ledger(audit_status_path(Path(workspace), metadata_dir))
        if not ledger:
            return {}
        statuses: dict[str, str] = {}
        for path in hit_paths:
            body = read_body_text(Path(workspace) / path)
            if body is None:
                continue
            status = ledger_status_for(path, body, ledger)
            if status:
                statuses[path] = status
        return statuses

    async def _frontmatter_flags(self, muscle_tier: dict) -> tuple[dict[str, str], dict[str, str]]:
        """Frontmatter flags of muscle hits: promotion aliases + stamped status.

        Usage recorded before promotion is keyed by the long-term digest path;
        the promoted file keeps that provenance in its ``promoted_from``
        frontmatter field, so the reflex tier still gets the full boost. The
        decorative ``verified_status`` stamp (wiped by every muscle_sync
        rewrite) is returned for visibility; weighting authority is the
        status ledger, not frontmatter.
        """
        hit_paths = sorted({hit["path"] for hit in muscle_tier.get("hits") or []})
        if not hit_paths:
            return {}, {}
        try:
            nodes = await self.file_store.get_nodes(hit_paths)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"[{self.name}] frontmatter lookup failed: {type(exc).__name__}: {exc}")
            return {}, {}
        aliases: dict[str, str] = {}
        statuses: dict[str, str] = {}
        for node in nodes or []:
            extras = getattr(node.front_matter, "model_extra", None) or {}
            promoted_from = str(extras.get("promoted_from") or "").strip()
            if promoted_from:
                aliases[node.path] = promoted_from
            stamped = str(extras.get("verified_status") or "").strip().lower()
            if stamped in VERIFIED_STATUS_VALUES:
                statuses[node.path] = stamped
        return aliases, statuses
