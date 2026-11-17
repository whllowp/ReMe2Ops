"""Cloud link repair: dangling detection, plan rules, gated stubs, step wiring."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileLink, FileNode
from reme.steps.ops.link_repair import (
    OpsLinkRepairStep,
    best_redirect_target,
    collect_links,
    find_dangling,
    plan_repairs,
    stub_page_path,
)


def _node(path: str, targets: list[str]) -> FileNode:
    return FileNode(
        path=path,
        st_mtime=0.0,
        links=[FileLink(source_path=path, target_path=target) for target in targets],
    )


KNOWN = {"digest/wiki/a.md", "digest/wiki/b.md", "digest/procedure/c.md", "digest/wiki/disk-cleanup.md"}


def test_collect_links_and_find_dangling():
    nodes = [
        _node("digest/wiki/a.md", ["digest/wiki/b.md", "daily/2026-11-01/src.md"]),
        _node("digest/procedure/c.md", ["digest/wiki/tyop.md", "digest/wiki/tyop.md"]),
        _node("digest/wiki/b.md", ["digest/wiki/a.md"]),
    ]
    links = collect_links(nodes)
    dangling = find_dangling(links, KNOWN)
    assert ("digest/wiki/a.md", "daily/2026-11-01/src.md") in dangling
    assert ("digest/procedure/c.md", "digest/wiki/tyop.md") in dangling  # deduped
    assert ("digest/wiki/b.md", "digest/wiki/a.md") not in dangling  # resolved edge


def test_best_redirect_target_threshold_and_ties():
    assert best_redirect_target("digest/wiki/disk-cleanup0.md", KNOWN, threshold=0.85) == "digest/wiki/disk-cleanup.md"
    assert best_redirect_target("digest/wiki/totally-unrelated-zzz.md", KNOWN, threshold=0.85) is None
    # ties resolve to the lexicographically first equally-good path
    known = {"digest/wiki/opt-a.md", "digest/wiki/opt-b.md"}
    assert best_redirect_target("digest/wiki/opt-x.md", known, threshold=0.5) == "digest/wiki/opt-a.md"


def test_stub_page_path_honors_path_targets():
    assert stub_page_path("digest/wiki/orphan-page.md", "digest/wiki") == "digest/wiki/orphan-page.md"
    assert stub_page_path("orphan-page", "digest/wiki") == "digest/wiki/orphan-page.md"


def test_plan_repairs_rules():
    dangling = [
        ("digest/wiki/a.md", "daily/2026-11-01/note.md"),  # daily target: intentional, skip
        ("digest/wiki/a.md", "digest/wiki/disk-cleanup0.md"),  # similar stem: redirect
        ("digest/wiki/b.md", "digest/wiki/orphan-page.md"),  # inbound 1: no stub yet
        ("digest/wiki/c.md", "digest/wiki/orphan-page.md"),  # raises inbound to 2
        ("digest/wiki/d.md", "digest/wiki/nothing-like-it.md"),  # no match, inbound 1: skip
    ]
    plans = plan_repairs(
        dangling,
        KNOWN,
        daily_prefix="daily",
        wiki_dir="digest/wiki",
        min_inbound_for_stub=2,
        max_repairs_per_run=5,
    )
    kinds = [plan["kind"] for plan in plans]
    assert kinds == ["redirect", "stub"]  # daily target and lonely orphan produce nothing
    redirect = plans[0]
    assert redirect["replacement"] == "digest/wiki/disk-cleanup.md"
    assert redirect["old"] == "[[digest/wiki/disk-cleanup0.md]]"
    assert redirect["new"] == "[[digest/wiki/disk-cleanup.md]]"
    stub = plans[1]
    assert stub["path"] == "digest/wiki/orphan-page.md"
    assert stub["inbound"] == ["digest/wiki/b.md", "digest/wiki/c.md"]
    assert "[[digest/wiki/b.md]]" in stub["content"]


def test_plan_repairs_caps_per_run():
    dangling = [
        (f"digest/wiki/s{i}.md", f"digest/wiki/disk-cleanup{i}.md")  # six redirectable targets
        for i in range(6)
    ]
    plans = plan_repairs(
        dangling,
        KNOWN,
        daily_prefix="daily",
        wiki_dir="digest/wiki",
        max_repairs_per_run=5,
    )
    assert len(plans) == 5
    assert all(plan["kind"] == "redirect" for plan in plans)


def _step(tmp_path: Path, nodes):
    store = Mock(spec=BaseFileStore, workspace_path=tmp_path)
    store.get_nodes = AsyncMock(return_value=nodes)
    return OpsLinkRepairStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
    )


@pytest.mark.asyncio
async def test_link_repair_step_redirects_and_keeps_daily_links(tmp_path):
    source = tmp_path / "digest" / "wiki" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "See [[digest/wiki/disk-cleanup0.md]] first; provenance [[daily/2026-11-01/n.md]]\n",
        encoding="utf-8",
    )
    nodes = [
        _node("digest/wiki/notes.md", ["digest/wiki/disk-cleanup0.md", "daily/2026-11-01/n.md"]),
        _node("digest/wiki/disk-cleanup.md", []),
    ]
    step = _step(tmp_path, nodes)

    async def fake_run_job(name, /, **kwargs):
        assert name == "edit"
        target = tmp_path / kwargs["path"]
        target.write_text(target.read_text(encoding="utf-8").replace(kwargs["old"], kwargs["new"]), encoding="utf-8")
        return Mock(success=True, answer="edited")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    response = await step(RuntimeContext())

    text = source.read_text(encoding="utf-8")
    assert "[[digest/wiki/disk-cleanup.md]]" in text  # redirected
    assert "[[daily/2026-11-01/n.md]]" in text  # daily provenance stays virtual
    meta = response.metadata["link_repair"]
    assert meta["redirects"] == 1 and meta["stubs"] == 0
    assert meta["dangling"] == 2
    assert response.success is True  # untouched default


@pytest.mark.asyncio
async def test_link_repair_step_writes_gated_stub_and_preserves_failure(tmp_path):
    """Two inbound links create a stub page; a preset failure state survives."""
    for name in ("digest/wiki/x.md", "digest/wiki/y.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"link to [[digest/wiki/orphan-page.md]] from {name}\n", encoding="utf-8")
    nodes = [
        _node("digest/wiki/x.md", ["digest/wiki/orphan-page.md"]),
        _node("digest/wiki/y.md", ["digest/wiki/orphan-page.md"]),
        _node("digest/wiki/other.md", []),
    ]
    step = _step(tmp_path, nodes)
    writes: list[dict] = []

    async def fake_run_job(name, /, **kwargs):
        writes.append(kwargs)
        assert name == "write"
        target = tmp_path / kwargs["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(kwargs["content"], name=kwargs["name"], description=kwargs["description"])
        target.write_text(frontmatter.dumps(post), encoding="utf-8")
        return Mock(success=True, answer="written")

    step.run_job = AsyncMock(side_effect=fake_run_job)
    context = RuntimeContext()
    context.response.success = False
    context.response.answer = "transitions step failed"

    response = await step(context)
    stub = tmp_path / "digest" / "wiki" / "orphan-page.md"
    assert stub.is_file()
    body = stub.read_text(encoding="utf-8")
    assert "2 page(s) link here" in body
    assert writes[0]["name"] == "orphan-page"
    assert response.success is False  # shared-Response contract: untouched
    assert response.answer == "transitions step failed"
    assert response.metadata["link_repair"]["stubs"] == 1


@pytest.mark.asyncio
async def test_link_repair_step_edit_failure_is_recorded_not_fatal(tmp_path):
    source = tmp_path / "digest" / "wiki" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text("See [[digest/wiki/disk-cleanup0.md]]\n", encoding="utf-8")
    nodes = [
        _node("digest/wiki/notes.md", ["digest/wiki/disk-cleanup0.md"]),
        _node("digest/wiki/disk-cleanup.md", []),
    ]
    step = _step(tmp_path, nodes)
    step.run_job = AsyncMock(return_value=Mock(success=False, answer="old text not found"))
    response = await step(RuntimeContext())

    meta = response.metadata["link_repair"]
    assert meta["redirects"] == 0
    assert meta["failed"] == [{"kind": "redirect", "source": "digest/wiki/notes.md", "error": "old text not found"}]
    assert response.success is True  # degraded, not failed
