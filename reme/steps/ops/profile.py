"""Device profile (SHARP §1): a dynamic memory bank with bounded context.

The profile is the tuple ⟨P, C, S⟩:
- ``P`` long-term profile — environment routines that keep holding (topology,
  alert mix, trusted runbooks), rewritten each round;
- ``C`` context state — what the device has been hitting lately versus P;
- ``S`` statistics — computed deterministically by ``build_statistics``.

Updates are the SHARP iteration: ``P_n = Agent(T, P_{n-1}, S_n)`` — each round
sees only the new records since the last cursor plus the *compressed previous
profile*, so the input stays flat as history grows (the paper's token-bound
dynamic update; ~91% saved versus full history at scale). Without an LLM the
step degrades to the deterministic S-only rendering. Every round writes a new
derived version with ``derived_from``/``model_version`` provenance (memery:
derived views are rebuildable).
"""

import asyncio
import datetime as _dt
import json
from pathlib import Path

import frontmatter

from ...components import R
from ..base_step import BaseStep
from .connector_client import default_device_id
from ..evolve._evolve import agent_reply_result_text
from ..evolve.dream.utils import llm_available, parse_structured_reply
from .usage import now_iso, read_usage_records, usage_file_path

PROFILE_CHARS = 4000  # bounded context: the compressed profile, not the history


def build_statistics(records: list[dict]) -> dict:
    """Deterministic S: hit distribution, outcomes, hour histogram."""
    solved = failed = 0
    paths: dict[str, int] = {}
    hours: dict[str, int] = {}
    for record in records:
        outcome = str(record.get("outcome") or "").strip().lower()
        if outcome not in {"solved", "failed"}:
            continue
        if outcome == "solved":
            solved += 1
        else:
            failed += 1
        path = str(record.get("path") or "").strip()
        if path:
            paths[path] = paths.get(path, 0) + 1
        hour = parse_ts(record.get("ts"))
        if hour is not None:
            bucket = f"{hour.hour:02d}"
            hours[bucket] = hours.get(bucket, 0) + 1
    top = sorted(paths.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return {
        "total": solved + failed,
        "solved": solved,
        "failed": failed,
        "success_rate": round(solved / max(solved + failed, 1), 4),
        "top_paths": [{"path": p, "hits": n} for p, n in top],
        "hour_histogram": dict(sorted(hours.items())),
    }


def parse_ts(raw):
    try:
        return _dt.datetime.fromisoformat(str(raw).strip())
    except (TypeError, ValueError):
        return None


STATS_MARKER = "## Recent statistics"


def render_profile_deterministic(P_prev: str, stats: dict, new_records: int) -> str:
    """S-only fallback: base profile + a REPLACED statistics block.

    Pure function of (previous base, latest stats): the stats section is
    regenerated each round instead of appended, so the profile stays bounded
    (SHARP's compressed-memory contract) rather than growing with history.
    """
    base = P_prev.strip()
    marker_index = base.find(STATS_MARKER)
    if marker_index >= 0:
        base = base[:marker_index].strip()
    lines = [base, "", STATS_MARKER + f" (+{new_records} events)", "",
             f"- events: {stats['total']} (solved {stats['solved']} / failed {stats['failed']}, "
             f"rate {stats['success_rate']})"]
    if stats["top_paths"]:
        lines.append("- most-used runbooks:")
        for item in stats["top_paths"][:5]:
            lines.append(f"  - {item['path']} x{item['hits']}")
    if stats["hour_histogram"]:
        lines.append(f"- activity hours: {json.dumps(stats['hour_histogram'])}")
    return "\n".join(lines).strip()[:PROFILE_CHARS]


def parse_profile_reply(text: str) -> str:
    """Extract the rewritten profile body from the agent reply."""
    data = parse_structured_reply(text, strict=True)
    profile = str(data.get("profile") or "").strip()
    if not profile:
        raise ValueError("reply missing 'profile'")
    return profile[:PROFILE_CHARS]


@R.register("ops_profile_step")
class OpsProfileStep(BaseStep):
    """One SHARP update round for a device profile (P/C/S), bounded context."""

    async def execute(self):
        assert self.context is not None
        workspace = self.file_store.workspace_path
        metadata_dir = str(self.config_value("metadata_dir"))
        device = str(self.context.get("device_id") or self.kwargs.get("device_id") or default_device_id())
        profile_rel = f"{metadata_dir}/ops_profile/{device}.md"
        profile_path: Path = workspace / profile_rel
        usage_path = usage_file_path(workspace, metadata_dir, cloud=True)

        previous = ""
        version = 0
        cursor = ""
        if profile_path.is_file():
            post = frontmatter.loads(profile_path.read_text(encoding="utf-8"))
            previous = post.content
            version = int(post.metadata.get("version") or 0)
            cursor = str(post.metadata.get("cursor_ts") or "")

        records = read_usage_records(usage_path)
        new_records = [r for r in records if str(r.get("ts") or "") > cursor and r.get("source_device") == device]
        if not new_records and previous:
            self.context.response.answer = f"Profile '{device}' v{version} up to date (no new events)"
            self.context.response.metadata["profile"] = {"device": device, "version": version, "updated": False}
            return self.context.response

        stats = build_statistics(new_records or records)
        derived_from = f"{profile_rel}#v{version}" if version else "seed"

        if llm_available(self):
            # SHARP two-stage update: long-term profile revision, then context
            # interpretation, both bounded by the compressed previous profile.
            update_timeout = float(self.kwargs.get("update_timeout", 90.0))
            try:
                system_prompt = self.prompt_format("profile_system_prompt")
                user_message = self.prompt_format(
                    "profile_user_message",
                    previous_profile=previous or "(empty — first round)",
                    stats_json=json.dumps(stats, ensure_ascii=False),
                )
                result = await asyncio.wait_for(
                    self.agent_wrapper.reply(user_message, system_prompt=system_prompt),
                    timeout=update_timeout,
                )
                profile_body = parse_profile_reply(agent_reply_result_text(result))
                stage = "agent"
            except Exception as exc:  # noqa: BLE001 - degrade to deterministic S
                self.logger.warning(f"[{self.name}] agent profile update failed: {exc}; falling back to statistics")
                profile_body = render_profile_deterministic(previous, stats, len(new_records))
                stage = "statistics"
        else:
            profile_body = render_profile_deterministic(previous, stats, len(new_records))
            stage = "statistics"

        version += 1
        post = frontmatter.Post(
            profile_body,
            device=device,
            version=version,
            derived_from=derived_from,
            model_version="ops-profile-v1",
            cursor_ts=max((str(r.get("ts") or "") for r in new_records), default=cursor),
            stats_total=stats["total"],
            updated_at=now_iso(),
        )
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        self.context.response.answer = (
            f"Profile '{device}' updated to v{version} ({stage} stage, {len(new_records)} new events, "
            f"{len(profile_body)} chars bounded)"
        )
        self.context.response.metadata["profile"] = {
            "device": device,
            "version": version,
            "stage": stage,
            "new_events": len(new_records),
            "chars": len(profile_body),
            "derived_from": derived_from,
            "path": profile_rel,
        }
        return self.context.response
