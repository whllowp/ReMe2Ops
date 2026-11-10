"""Reflex card generation: prepend a conditioned-reflex section to promoted SOPs."""

import json
from unittest.mock import Mock

import frontmatter
import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.reflex_card import (
    OpsReflexCardStep,
    ReflexCard,
    REFLEX_CARD_HEADING,
    render_card_section,
)

CARD_JSON = json.dumps(
    {"trigger": "磁盘利用率告警 >90%", "actions": ["du -sh /var/log/*", "journalctl --vacuum-size=2G"], "pitfall": "删除正在写入的日志"},
    ensure_ascii=False,
)


class FakeWrapper(BaseAgentWrapper):
    """Agent wrapper returning a canned completed reply."""

    def __init__(self, reply_text: str = CARD_JSON, error: Exception | None = None):
        super().__init__()
        self.reply_text = reply_text
        self.error = error
        self.calls = []

    async def reply(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        if self.error is not None:
            raise self.error
        return {"status": "completed", "error": None, "result": self.reply_text}


def _step(tmp_path, wrapper, **kwargs):
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    return OpsReflexCardStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
        agent_wrapper=wrapper,
        **kwargs,
    )


def _write_sop(tmp_path, *, body: str | None = None) -> str:
    dst = "digest/muscle/disk-space-cleanup.md"
    path = tmp_path / dst
    path.parent.mkdir(parents=True, exist_ok=True)
    content = body if body is not None else "## Trigger\n磁盘告警。\n\n## Steps\n1. du -sh\n"
    path.write_text(
        frontmatter.dumps(frontmatter.Post(content, name="disk-space-cleanup", description="d", memory_type="muscle")),
        encoding="utf-8",
    )
    return dst


def test_render_card_section():
    card = ReflexCard(trigger="t", actions=["a", "b"], pitfall="p")
    section = render_card_section(card)
    assert section.startswith(REFLEX_CARD_HEADING)
    assert "**Trigger**: t" in section
    assert "1. a" in section and "2. b" in section
    assert "**Pitfall**: p" in section


@pytest.mark.asyncio
async def test_reflex_card_prepended_after_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.reflex_card.llm_available", lambda step: True)
    dst = _write_sop(tmp_path)
    step = _step(tmp_path, FakeWrapper(), reflex_card=True)
    response = await step(RuntimeContext(promoted={"from": "digest/procedure/x.md", "to": dst}))

    assert response.metadata["reflex_card"]["actions"] == 2
    post = frontmatter.loads((tmp_path / dst).read_text(encoding="utf-8"))
    assert post.content.startswith(REFLEX_CARD_HEADING)
    assert "journalctl --vacuum-size=2G" in post.content
    assert "## Trigger" in post.content  # original runbook intact below the card
    assert post.metadata["memory_type"] == "muscle"
    # the prompt carried the runbook body
    assert "du -sh" in step.agent_wrapper.calls[0][0]


@pytest.mark.asyncio
async def test_reflex_card_disabled_is_noop(tmp_path):
    dst = _write_sop(tmp_path)
    step = _step(tmp_path, FakeWrapper())
    response = await step(RuntimeContext(promoted={"from": "a", "to": dst}))
    assert "reflex_card" not in response.metadata
    assert REFLEX_CARD_HEADING not in (tmp_path / dst).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reflex_card_without_promotion_is_noop(tmp_path):
    _write_sop(tmp_path)
    step = _step(tmp_path, FakeWrapper(), reflex_card=True)
    response = await step(RuntimeContext())
    assert response.metadata["reflex_card"]["skipped"] == "no promotion in this job"


@pytest.mark.asyncio
async def test_reflex_card_skips_existing_card(tmp_path):
    dst = _write_sop(tmp_path, body=f"{REFLEX_CARD_HEADING}\n\n1. already here\n\n## Trigger\nx\n")
    step = _step(tmp_path, FakeWrapper(), reflex_card=True)
    response = await step(RuntimeContext(promoted={"to": dst}))
    assert response.metadata["reflex_card"]["skipped"] == "card already present"


@pytest.mark.asyncio
async def test_reflex_card_survives_llm_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.reflex_card.llm_available", lambda step: True)
    dst = _write_sop(tmp_path)
    step = _step(tmp_path, FakeWrapper(error=RuntimeError("llm down")), reflex_card=True)
    response = await step(RuntimeContext(promoted={"to": dst}))
    assert "llm down" in response.metadata["reflex_card"]["skipped"]
    # promotion output untouched
    assert REFLEX_CARD_HEADING not in (tmp_path / dst).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reflex_card_survives_unparseable_reply(tmp_path, monkeypatch):
    monkeypatch.setattr("reme.steps.ops.reflex_card.llm_available", lambda step: True)
    dst = _write_sop(tmp_path)
    step = _step(tmp_path, FakeWrapper(reply_text="no json here"), reflex_card=True)
    response = await step(RuntimeContext(promoted={"to": dst}))
    assert "Invalid structured reply" in response.metadata["reflex_card"]["skipped"]
    assert REFLEX_CARD_HEADING not in (tmp_path / dst).read_text(encoding="utf-8")
