"""Decision-model backends: openai_chat LLM-as-judge and agentscope model_class."""

import json
from types import SimpleNamespace

import pytest

from reme.components.as_decision_model.agentscope import AgentScopeAsDecisionModel
from reme.components.as_decision_model.openai_chat import (
    OpenAIChatAsDecisionModel,
    _extract_json_object,
    _parse_answer,
    _render_user_prompt,
)


class _FakeCompletions:
    def __init__(self, replies, calls):
        self._replies = list(replies)
        self._calls = calls

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=None)
        _FakeAsyncOpenAI.instances.append(self)


def _decision(credential=None, model="mock-decider", **kwargs):
    return OpenAIChatAsDecisionModel(credential=credential or {"api_key": "test-key"}, model=model, **kwargs)


def test_render_user_prompt_covers_question_types():
    prompt = _render_user_prompt(
        {"candidates": [{"path": "a.md"}]},
        {
            "q_bin": {"type": "binary", "instructions": "judge", "criteria": {"true": "yes case", "false": "no case"}},
            "q_choice": {"type": "choice", "criteria": {"fast": None, "slow": None}},
            "q_score": {"type": "score", "criteria": ["low", "mid", "high"]},
        },
    )
    assert '"q_bin" (binary)' in prompt
    assert "TRUE means: yes case" in prompt
    assert "alternatives: fast; slow" in prompt
    assert "rubric: 0=low; 1=mid; 2=high" in prompt
    assert "a.md" in prompt


def test_extract_json_object_accepts_raw_and_fenced():
    assert _extract_json_object('{"answers": {"a": {"probability": 1}}}') == {"a": {"probability": 1}}
    fenced = 'prose\n```json\n{"answers": {"a": {"probability": 0}}}\n```\n'
    assert _extract_json_object(fenced) == {"a": {"probability": 0}}
    bare_number = {"a": 0.7}
    assert _extract_json_object(json.dumps(bare_number)) == bare_number
    with pytest.raises(ValueError):
        _extract_json_object("not json at all")


def test_parse_answer_shapes():
    binary = _parse_answer({"probability": 0.9}, {"type": "binary"})
    assert binary.type == "binary_answer" and binary.probability == pytest.approx(0.9)
    bare = _parse_answer(0.4, {"type": "binary"})
    assert bare.probability == pytest.approx(0.4)
    choice = _parse_answer({"choice": "fast", "confidence": 0.8}, {"type": "choice", "criteria": {"fast": None}})
    assert choice.choice == "fast" and choice.confidence == pytest.approx(0.8)
    score = _parse_answer({"score": 2}, {"type": "score", "criteria": ["low", "mid", "high"]})
    assert score.score == 2
    with pytest.raises(ValueError):
        _parse_answer({"probability": 1.5}, {"type": "binary"})
    with pytest.raises(ValueError):
        _parse_answer({"choice": "bogus"}, {"type": "choice", "criteria": {"fast": None}})
    with pytest.raises(ValueError):
        _parse_answer({"score": 5}, {"type": "score", "criteria": ["low"]})


@pytest.mark.asyncio
async def test_openai_chat_call_parses_typed_answers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(
                ['```json\n{"answers": {"m": {"probability": 0.92}, "e": {"probability": 0.55}}}\n```'],
                calls,
            )),
            close=lambda: None,
        ),
    )
    decision = _decision()
    response = await decision(
        state={"candidates": [{"path": "digest/procedure/a.md"}]},
        questions={"m": {"type": "binary"}, "e": {"type": "binary"}},
    )
    assert response.content["m"].probability == pytest.approx(0.92)
    assert response.content["e"].type == "binary_answer"
    assert "digest/procedure/a.md" in calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_openai_chat_retries_transport_then_succeeds(monkeypatch):
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(
                [ConnectionError("boom"), '{"answers": {"m": {"probability": 0.8}}}'],
                [],
            )),
            close=lambda: None,
        ),
    )
    decision = _decision(max_retries=1, retry_delay=0.0)
    response = await decision(state="state text", questions={"m": {"type": "binary"}})
    assert response.content["m"].probability == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_openai_chat_unusable_reply_raises(monkeypatch):
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(["garbage", "still garbage"], [])),
            close=lambda: None,
        ),
    )
    decision = _decision(max_retries=1, retry_delay=0.0)
    with pytest.raises(ValueError):
        await decision(state="state", questions={"m": {"type": "binary"}})


@pytest.mark.asyncio
async def test_openai_chat_requires_api_key():
    decision = OpenAIChatAsDecisionModel(credential={"api_key": ""}, model="m")
    with pytest.raises(ValueError, match="DECISION_MODEL_API_KEY"):
        await decision(state="state", questions={"m": {"type": "binary"}})


@pytest.mark.asyncio
async def test_openai_chat_requires_model():
    decision = OpenAIChatAsDecisionModel(credential={"api_key": "k"}, model=None)
    with pytest.raises(ValueError, match="DECISION_MODEL_NAME"):
        await decision(state="state", questions={"m": {"type": "binary"}})


@pytest.mark.asyncio
async def test_agentscope_backend_selects_model_class(monkeypatch):
    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

    class FakeJevModel:
        pass

    classifier_module = SimpleNamespace(
        ClassifierQuestion=dict,
        JevClassifierModel=FakeJevModel,
        OpenAIClassifierModel=FakeOpenAIModel,
    )
    credential_calls = []
    credential_module = SimpleNamespace(
        TypeSafeCredential=lambda **kwargs: credential_calls.append(("typesafe", kwargs)) or SimpleNamespace(name="typesafe"),
        OpenAICredential=lambda **kwargs: credential_calls.append(("openai", kwargs)) or SimpleNamespace(name="openai"),
    )

    def fake_import(name):
        return {"agentscope.classifier": classifier_module, "agentscope.credential": credential_module}[name]

    monkeypatch.setattr("reme.components.as_decision_model.agentscope.import_module", fake_import)
    component = AgentScopeAsDecisionModel(credential={"api_key": "k"}, model="m", model_class="openai")

    question_cls, model_cls = component._load_decision_model(component.options.model_class)
    assert model_cls is classifier_module.OpenAIClassifierModel

    component._create_model(model_cls, {"api_key": "k", "base_url": None}, "openai")
    assert isinstance(component._model, FakeOpenAIModel)
    assert credential_calls == [("openai", {"api_key": "k", "base_url": None})]
