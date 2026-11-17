"""Cloud-side link repair: the second dreaming pass fixes dangling wikilinks.

Paper component ③ (dream gap repair: a second dreaming pass re-linked nine
dangling links and added three missing pages including their index). Two
hard-won constraints shape this step:

- Everything is deterministic — no LLM — and stub creation is gated by an
  inbound-link minimum, because a wiki can inflate until the map is bigger
  than the territory (the paper's Borges warning).
- Dangling targets under ``daily/`` are intentional short-term → long-term
  provenance links (design doc §1: digest ``## Sources`` wikilinks point back
  at edge daily notes as virtual nodes) and are never "repaired".

Shared-Response contract: this step runs inside ``ops_maintenance_cron`` after
decay/profile/transitions, which share one Response and whose final
``success``/``answer`` feed the job status — so only
``metadata["link_repair"]`` is written here.
"""

from difflib import SequenceMatcher
from pathlib import PurePosixPath

from ...components import R
from ..base_step import BaseStep


def collect_links(nodes) -> list[tuple[str, str]]:
    """(source, target) edges from indexed nodes; same data source as traverse."""
    links: list[tuple[str, str]] = []
    for node in nodes or []:
        source = str(getattr(node, "path", "") or "").strip()
        for link in getattr(node, "links", None) or []:
            target = str(getattr(link, "target_path", "") or "").strip()
            if source and target:
                links.append((source, target))
    return links


def find_dangling(links: list[tuple[str, str]], known_paths: set[str]) -> list[tuple[str, str]]:
    """Edges whose literal target is not an indexed path, deduped and sorted."""
    return sorted({(s, t) for s, t in links if t not in known_paths and s != t})


def best_redirect_target(target: str, known_paths: set[str], *, threshold: float = 0.85) -> str | None:
    """Closest indexed path by stem similarity (≥ threshold), or None.

    Ties resolve to the lexicographically first path so plans are stable.
    """
    stem = PurePosixPath(target.replace("\\", "/")).stem.lower()
    scored = [
        (SequenceMatcher(None, stem, PurePosixPath(path).stem.lower()).ratio(), path)
        for path in sorted(known_paths)
    ]
    scored = [item for item in scored if item[0] >= threshold]
    if not scored:
        return None
    best_score = max(score for score, _ in scored)
    return next(path for score, path in scored if score == best_score)


def stub_page_path(target: str, wiki_dir: str) -> str:
    """Where a stub for a dangling target should live.

    Path-shaped markdown targets are honored as-is; bare names land in the
    wiki bucket (the digest index bucket) under their stem.
    """
    normalized = target.replace("\\", "/")
    if normalized.endswith(".md") and "/" in normalized:
        return normalized
    stem = PurePosixPath(normalized).stem.strip() or "index"
    return f"{wiki_dir.strip('/')}/{stem}.md"


def render_stub_page(target: str, sources: list[str]) -> str:
    """Deterministic stub body: what links here, nothing invented."""
    stem = PurePosixPath(target.replace("\\", "/")).stem
    lines = [f"# {stem}", ""]
    lines.append(f"Stub page created by ops_link_repair because {len(sources)} page(s) link here:")
    lines.append("")
    for source in sources:
        lines.append(f"- [[{source}]]")
    lines.extend(["", "Expand this page or retarget the inbound links; stubs are never LLM-filled."])
    return "\n".join(lines) + "\n"


def plan_repairs(
    dangling: list[tuple[str, str]],
    known_paths: set[str],
    *,
    daily_prefix: str,
    wiki_dir: str,
    min_inbound_for_stub: int = 2,
    max_repairs_per_run: int = 5,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Deterministic repair plans: redirects first, gated stubs second, capped.

    - daily-prefix targets are skipped (intentional virtual provenance nodes);
    - a target with a similar existing page becomes one redirect plan per
      target (``old``/``new`` are exact ``[[...]]`` literals for the edit job);
    - otherwise, when at least ``min_inbound_for_stub`` distinct sources link
      to it, one write plan creates a stub page;
    - everything else stays dangling — no inflation.
    """
    daily_prefix = daily_prefix.strip("/")
    prefix_with_slash = f"{daily_prefix}/" if daily_prefix else ""
    inbound: dict[str, set[str]] = {}
    for source, target in dangling:
        inbound.setdefault(target, set()).add(source)

    plans: list[dict] = []
    for target in sorted(inbound):
        if len(plans) >= max_repairs_per_run:
            break
        if prefix_with_slash and target.startswith(prefix_with_slash):
            continue
        sources = sorted(inbound[target])
        replacement = best_redirect_target(target, known_paths, threshold=similarity_threshold)
        if replacement:
            plans.append(
                {
                    "kind": "redirect",
                    "target": target,
                    "replacement": replacement,
                    "sources": sources,
                    "old": f"[[{target}]]",
                    "new": f"[[{replacement}]]",
                },
            )
            continue
        if len(sources) >= max(int(min_inbound_for_stub), 1):
            stub_path = stub_page_path(target, wiki_dir)
            plans.append(
                {
                    "kind": "stub",
                    "target": target,
                    "path": stub_path,
                    "name": PurePosixPath(stub_path).stem,
                    "inbound": sources,
                    "description": f"Stub created by ops_link_repair: {len(sources)} inbound link(s)",
                    "content": render_stub_page(target, sources),
                },
            )
    return plans[: max(int(max_repairs_per_run), 1)]


@R.register("ops_link_repair_step")
class OpsLinkRepairStep(BaseStep):
    """Redirect dangling wikilinks and stub well-linked missing pages.

    Reads the file graph (``get_nodes()``), plans repairs deterministically,
    and applies them with the ``edit``/``write`` jobs. Job success accounting
    stays with the cron chain: only ``metadata["link_repair"]`` is written.
    """

    async def execute(self):
        assert self.context is not None
        nodes = await self.file_store.get_nodes()
        links = collect_links(nodes)
        known = {str(getattr(node, "path", "") or "").strip() for node in nodes or []}
        known.discard("")

        digest_dir = str(self.config_value("digest_dir")).strip("/")
        daily_prefix = str(self.config_value("daily_dir")).strip("/")
        wiki_dir = str(self.kwargs.get("wiki_dir") or f"{digest_dir}/wiki").strip("/")
        min_inbound = max(int(self.kwargs.get("min_inbound_for_stub", 2)), 1)
        max_repairs = max(int(self.kwargs.get("max_repairs_per_run", 5)), 1)
        threshold = float(self.kwargs.get("similarity_threshold", 0.85))

        dangling = find_dangling(links, known)
        plans = plan_repairs(
            dangling,
            known,
            daily_prefix=daily_prefix,
            wiki_dir=wiki_dir,
            min_inbound_for_stub=min_inbound,
            max_repairs_per_run=max_repairs,
            similarity_threshold=threshold,
        )

        applied: list[dict] = []
        failed: list[dict] = []
        redirects = stubs = 0
        for plan in plans:
            if plan["kind"] == "redirect":
                for source in plan["sources"]:
                    try:
                        response = await self.run_job("edit", path=source, old=plan["old"], new=plan["new"])
                    except Exception as exc:  # noqa: BLE001 - one failed edit skips, never fatal
                        failed.append({"kind": "redirect", "source": source, "error": str(exc)})
                        continue
                    if response.success:
                        applied.append({"kind": "redirect", "source": source, "old": plan["old"], "new": plan["new"]})
                        redirects += 1
                    else:
                        failed.append({"kind": "redirect", "source": source, "error": str(response.answer)})
            else:
                try:
                    response = await self.run_job(
                        "write",
                        path=plan["path"],
                        name=plan["name"],
                        description=plan["description"],
                        content=plan["content"],
                    )
                except Exception as exc:  # noqa: BLE001
                    failed.append({"kind": "stub", "path": plan["path"], "error": str(exc)})
                    continue
                if response.success:
                    applied.append({"kind": "stub", "path": plan["path"], "target": plan["target"]})
                    stubs += 1
                else:
                    failed.append({"kind": "stub", "path": plan["path"], "error": str(response.answer)})

        self.context.response.metadata["link_repair"] = {
            "dangling": len(dangling),
            "plans": plans,
            "applied": applied,
            "failed": failed,
            "redirects": redirects,
            "stubs": stubs,
        }
        return self.context.response
