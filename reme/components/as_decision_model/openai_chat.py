"""OpenAI-compatible chat backend for the decision model (LLM-as-judge).

Plugs any OpenAI-compatible chat-completions endpoint (self-hosted decision
models such as Laya, Qwen, DeepSeek, or a vLLM server) into the
``as_decision_model`` abstraction without a provider-specific SDK: state and
questions are rendered into a strict-JSON prompt, and the reply is parsed into
the same typed answer shapes the agentscope classifier backend returns.
"""

import asyncio
import json
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from ..component_registry import R
from .base_as_decision_model import BaseAsDecisionModel

_SYSTEM_PROMPT = """You are a decision engine. You receive a state and named questions.
Answer EVERY question strictly from the state; never invent facts.
Binary questions take a probability in [0, 1] that the TRUE criteria hold.
Choice questions pick exactly one named alternative; score questions pick one
rubric level as a 0-based integer.
Respond with ONLY a JSON object, no prose, in this exact shape:
{"answers": {"<question name>": {"probability": 0.0}}}          // binary
{"answers": {"<question name>": {"choice": "<alternative>", "confidence": 0.0}}}   // choice
{"answers": {"<question name>": {"score": 0, "confidence": 0.0}}}                  // score
Use 0.5 probability / low confidence when the state is genuinely ambiguous."""


class _ChatOptions(BaseModel):
    """Validate transport options without touching the provider client."""

    model_config = ConfigDict(extra="forbid")

    # None is accepted at construction so an unconfigured env (DECISION_MODEL_NAME
    # unset) still builds; the call itself fails with an actionable error.
    model: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    retry_delay: float = Field(default=0.5, ge=0, allow_inf_nan=False)


def _question_block(name: str, question: Any) -> str:
    """Render one question into prompt lines understood by any chat model."""
    if not isinstance(question, dict):
        return f'- "{name}": {json.dumps(question, ensure_ascii=False)}'
    qtype = str(question.get("type") or "binary").strip().lower()
    lines = [f'- "{name}" ({qtype})']
    instructions = str(question.get("instructions") or "").strip()
    if instructions:
        lines.append(f"  instructions: {instructions}")
    criteria = question.get("criteria")
    if qtype == "binary" and isinstance(criteria, dict):
        if criteria.get("true"):
            lines.append(f'  TRUE means: {criteria["true"]}')
        if criteria.get("false"):
            lines.append(f'  FALSE means: {criteria["false"]}')
    elif qtype == "choice" and isinstance(criteria, dict):
        options = "; ".join(str(key) for key in criteria)
        lines.append(f"  alternatives: {options}")
    elif qtype == "score" and isinstance(criteria, list):
        levels = "; ".join(f"{index}={level}" for index, level in enumerate(criteria))
        lines.append(f"  rubric: {levels}")
    return "\n".join(lines)


def _render_user_prompt(state: str | dict, questions: Mapping[str, Any]) -> str:
    """Build the user message carrying the state and every question."""
    if isinstance(state, dict):
        state_text = json.dumps(state, ensure_ascii=False)
    else:
        state_text = state
    blocks = "\n".join(_question_block(name, question) for name, question in questions.items())
    return f"State:\n{state_text}\n\nQuestions (answer every one):\n{blocks}"


def _extract_json_object(text: str) -> dict:
    """Parse the JSON object from a reply, tolerating one fenced block."""
    candidates = [text.strip()]
    for marker in ("```json", "```"):
        if marker in text:
            for chunk in text.split(marker):
                stripped = chunk.strip()
                if stripped.startswith("{"):
                    candidates.insert(0, stripped.rstrip("`").strip())
            break
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            answers = data.get("answers")
            return answers if isinstance(answers, dict) else data
    raise ValueError(f"decision reply is not a JSON object: {text[:200]!r}")


def _parse_answer(raw: Any, question: Any) -> Any:
    """Convert one raw answer into the classifier answer shape."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = {"probability": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid answer payload: {raw!r}")
    qtype = str(question.get("type") or "binary").strip().lower() if isinstance(question, dict) else "binary"
    if qtype == "binary":
        probability = raw.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"binary answer needs a probability in [0, 1], got {probability!r}")
        return SimpleNamespace(type="binary_answer", probability=float(probability))
    confidence = raw.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
    if qtype == "choice":
        choice = str(raw.get("choice") or "")
        criteria = question.get("criteria") if isinstance(question, dict) else None
        if criteria is not None and choice not in criteria:
            raise ValueError(f"choice {choice!r} is not one of the alternatives")
        return SimpleNamespace(type="choice_answer", choice=choice, confidence=confidence)
    if qtype == "score":
        score = raw.get("score")
        criteria = question.get("criteria") if isinstance(question, dict) else None
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"score answer needs an integer score, got {score!r}")
        if isinstance(criteria, list) and not 0 <= score < len(criteria):
            raise ValueError(f"score {score} outside the rubric range [0, {len(criteria) - 1}]")
        return SimpleNamespace(type="score_answer", score=score, confidence=confidence)
    raise ValueError(f"unsupported question type: {qtype!r}")


@R.register("openai_chat")
class OpenAIChatAsDecisionModel(BaseAsDecisionModel):
    """LLM-as-judge over any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        credential: dict | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.credential = dict(credential or {})
        self.options = _ChatOptions(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._client: openai.AsyncOpenAI | None = None

    def _build_client(self) -> openai.AsyncOpenAI:
        """Construct the client lazily; an empty key fails with a clear error."""
        api_key = self.credential.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Decision model inference requires DECISION_MODEL_API_KEY or credential.api_key.")
        if not self.options.model:
            raise ValueError("Decision model inference requires DECISION_MODEL_NAME or credential.model.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.credential.get("base_url") or None,
            timeout=self.options.timeout,
            max_retries=self.options.max_retries,
        )

    async def _close(self) -> None:
        """Close the underlying client if it was built."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __call__(self, state: str | dict, questions: Mapping[str, Any]) -> Any:
        """Render state+questions into one chat call and parse typed answers."""
        if not isinstance(state, (str, dict)) or (isinstance(state, str) and not state.strip()):
            raise ValueError("Decision state must be non-empty text or a JSON object.")
        if isinstance(state, dict):
            json.dumps(state, allow_nan=False)
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("Decision questions must be a non-empty mapping.")

        if self._client is None:
            self._client = self._build_client()
        prompt = _render_user_prompt(state, questions)
        last_error: Exception | None = None
        for attempt in range(self.options.max_retries + 1):
            if attempt:
                await asyncio.sleep(self.options.retry_delay)
            try:
                completion = await self._client.chat.completions.create(
                    model=self.options.model,
                    temperature=self.options.temperature,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = completion.choices[0].message.content or ""
                answers = _extract_json_object(text)
                return SimpleNamespace(
                    model=self.options.model,
                    content={name: _parse_answer(answers.get(name), question) for name, question in questions.items()},
                    usage=None,
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = ValueError(f"decision reply unusable: {exc}")
            except Exception as exc:  # noqa: BLE001 - transport errors are retryable
                last_error = exc
        raise last_error if last_error is not None else RuntimeError("decision call failed")
