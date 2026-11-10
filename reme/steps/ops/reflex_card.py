"""Generate a compact reflex card on top of freshly promoted muscle SOPs.

A muscle SOP should read like a conditioned reflex: trigger, immediate
actions, the one pitfall. The full runbook stays untouched below the card,
so fidelity is never lost — and every failure path here skips gracefully,
because promotion itself has already happened.
"""

import asyncio

import frontmatter
from pydantic import BaseModel, Field

from ...components import R
from ..base_step import BaseStep
from ..evolve._evolve import agent_reply_result_text
from ..evolve.dream.utils import llm_available, parse_structured_reply

REFLEX_CARD_HEADING = "## Reflex Card"


class ReflexCard(BaseModel):
    """Compact conditioned-reflex card generated for one promoted muscle SOP."""

    trigger: str = Field(description="One line: the signal that should fire this reflex.")
    actions: list[str] = Field(
        default_factory=list,
        description="3-8 imperative steps in execution order, each one line.",
    )
    pitfall: str = Field(default="", description="The single mistake that ruins the outcome, if any.")


def render_card_section(card: ReflexCard) -> str:
    """Render the card as a Markdown section block."""
    lines = [REFLEX_CARD_HEADING, "", f"**Trigger**: {card.trigger}", ""]
    for index, action in enumerate(card.actions, start=1):
        lines.append(f"{index}. {action}")
    if card.pitfall:
        lines.extend(["", f"**Pitfall**: {card.pitfall}"])
    return "\n".join(lines).strip() + "\n"


@R.register("ops_reflex_card_step")
class OpsReflexCardStep(BaseStep):
    """Prepend a generated reflex card to the SOP promoted in this job.

    Enabled via the ``reflex_card`` kwarg; without an LLM, without a
    ``context["promoted"]`` entry, or when the card already exists, this is a
    no-op that records the skip reason in metadata.
    """

    async def execute(self):
        assert self.context is not None
        if not bool(self.kwargs.get("reflex_card", False)):
            return self.context.response
        promoted = self.context.get("promoted") or {}
        dst = str(promoted.get("to") or "")
        if not dst:
            self.context.response.metadata["reflex_card"] = {"skipped": "no promotion in this job"}
            return self.context.response

        workspace = self.file_store.workspace_path
        file_path = workspace / dst
        if not file_path.is_file():
            self.context.response.metadata["reflex_card"] = {"skipped": "promoted file missing", "path": dst}
            return self.context.response

        post = frontmatter.loads(file_path.read_text(encoding="utf-8"))
        if REFLEX_CARD_HEADING in post.content:
            self.context.response.metadata["reflex_card"] = {"skipped": "card already present", "path": dst}
            return self.context.response
        if not llm_available(self):
            self.context.response.metadata["reflex_card"] = {"skipped": "no llm configured", "path": dst}
            return self.context.response

        card_timeout = float(self.kwargs.get("card_timeout", 60.0))
        system_prompt = self.prompt_format("reflex_card_system_prompt", heading=REFLEX_CARD_HEADING)
        user_message = self.prompt_format("reflex_card_user_message", path=dst, body=post.content[:12000])
        try:
            result = await asyncio.wait_for(
                self.agent_wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=card_timeout,
            )
            card = ReflexCard.model_validate(parse_structured_reply(agent_reply_result_text(result), strict=True))
        except Exception as exc:  # noqa: BLE001 - a card failure never undoes promotion
            self.context.response.metadata["reflex_card"] = {
                "skipped": f"{type(exc).__name__}: {exc}",
                "path": dst,
            }
            self.logger.warning(f"[{self.name}] reflex card generation failed: {type(exc).__name__}: {exc}")
            return self.context.response

        if not card.actions:
            self.context.response.metadata["reflex_card"] = {"skipped": "empty actions", "path": dst}
            return self.context.response
        post.content = f"{render_card_section(card)}\n\n{post.content}"
        file_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        self.context.response.metadata["reflex_card"] = {
            "path": dst,
            "actions": len(card.actions),
            "trigger": card.trigger,
        }
        return self.context.response
