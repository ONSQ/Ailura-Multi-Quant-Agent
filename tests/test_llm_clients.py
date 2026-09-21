"""Provider clients are exercised with fake SDK objects (no network)."""

from types import SimpleNamespace

from ailura.agents.schemas import ReflectionLesson
from ailura.llm.anthropic_client import AnthropicLLM
from ailura.llm.base import LLMRequest
from ailura.llm.openai_client import OpenAICompatibleLLM

LESSON = ReflectionLesson(verdict="correct", what_the_outcome_shows="x", thesis_component_affected="y", lesson="z", tags=["t"])


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = self

    def parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=LESSON.model_dump_json())],
            parsed_output=LESSON, stop_reason="end_turn", model=kw["model"],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3),
        )

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")], stop_reason="end_turn", model=kw["model"], usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_anthropic_client_shapes_request():
    fake = FakeAnthropic()
    llm = AnthropicLLM(model="claude-opus-5", effort="high", client=fake)
    r = llm.complete(LLMRequest(role="reflector", system="s", user="u", schema=ReflectionLesson))
    assert r.parsed == LESSON and r.usage.cache_read_tokens == 3 and r.model == "claude-opus-5"
    kw = fake.calls[0]
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "high"}
    assert kw["output_format"] is ReflectionLesson and kw["extra_body"] == {"fallbacks": "default"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    r2 = llm.complete(LLMRequest(role="x", system="s", user="u"))
    assert r2.text == "hello" and r2.parsed is None


def test_anthropic_refusal_raises():
    import pytest

    from ailura.llm.base import LLMRefusalError

    class Refuse(FakeAnthropic):
        def parse(self, **kw):
            return SimpleNamespace(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="x"), model="m", usage=None)

    with pytest.raises(LLMRefusalError):
        AnthropicLLM(client=Refuse()).complete(LLMRequest(role="r", system="s", user="u", schema=ReflectionLesson))


class FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content="```json\n" + LESSON.model_dump_json() + "\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model=kw["model"], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2))


def test_openai_client_parses_fenced_json():
    fake = FakeOpenAI()
    llm = OpenAICompatibleLLM(model="gpt-x", client=fake)
    r = llm.complete(LLMRequest(role="reflector", system="s", user="u", schema=ReflectionLesson))
    assert r.parsed == LESSON and r.usage.input_tokens == 7
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in fake.calls[0]["messages"][0]["content"]
