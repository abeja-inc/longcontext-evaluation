import sys
import types
from logging import Logger
from typing import Any
from unittest.mock import MagicMock

import pytest


# Test environment may not have optional OpenAI dependency installed.
if "openai" not in sys.modules:

    class _OpenAI:
        pass

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _OpenAI
    sys.modules["openai"] = fake_openai

if "openai.types" not in sys.modules:
    sys.modules["openai.types"] = types.ModuleType("openai.types")

if "openai.types.responses" not in sys.modules:
    fake_responses = types.ModuleType("openai.types.responses")

    class _Response:
        pass

    fake_responses.Response = _Response
    sys.modules["openai.types.responses"] = fake_responses

# Test environment may not have optional tokenizer dependency installed.
if "tiktoken" not in sys.modules:
    sys.modules["tiktoken"] = types.SimpleNamespace(
        encoding_for_model=lambda *_args, **_kwargs: None,
        get_encoding=lambda *_args, **_kwargs: None,
    )

from llm_inference.data import Conversation, Message, Prompt, Response
from llm_inference.openai_api import OpenAIGenerator


class DummyTokenizer:
    def __init__(self) -> None:
        self.encode_inputs: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encode_inputs.append(text)
        return list(range(len(text)))


class WeirdConversation(Conversation):
    @property
    def prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": "abc",
                "name": "alice",
                "tool_call_id": 42,
                "ignored": None,
            },
            {"role": "assistant", "content": "z"},
        ]


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def generator(
    monkeypatch: pytest.MonkeyPatch, dummy_tokenizer: DummyTokenizer
) -> OpenAIGenerator:
    monkeypatch.setattr(
        "llm_inference.openai_api.tiktoken.encoding_for_model",
        lambda _model_name: dummy_tokenizer,
    )
    return OpenAIGenerator(
        client=MagicMock(),
        model_name="gpt-4o-mini",
        max_context_length=32,
        max_output_tokens=8,
        logger=MagicMock(spec=Logger),
    )


def test_count_tokens_prompt_uses_tokenizer_encode(generator: OpenAIGenerator) -> None:
    count = generator._count_tokens(Prompt(prompt="hello"))

    assert count == 5


def test_count_tokens_conversation_counts_overhead_special_fields_and_footer(
    generator: OpenAIGenerator,
) -> None:
    count = generator._count_tokens(WeirdConversation(messages=[]))

    # 2 messages * tokens_per_message(3) +
    # role/content/name/tool_call_id/content token lengths + tokens_per_name(1) + footer(3)
    assert count == (6 + 4 + 3 + 5 + 2 + 1 + 9 + 1 + 3)


def test_count_tokens_raises_type_error_for_unsupported_input(
    generator: OpenAIGenerator,
) -> None:
    with pytest.raises(TypeError, match="Unsupported input type"):
        generator._count_tokens("not-supported")  # type: ignore[arg-type]


def test_call_response_api_calls_openai_and_builds_response_with_usage_and_input_metadata(
    generator: OpenAIGenerator,
) -> None:
    prompt = Prompt(prompt="hello", metadata={"trace_id": "abc"})
    usage = MagicMock()
    usage.model_dump.return_value = {"input_tokens": 1, "output_tokens": 2}
    api_response = MagicMock(output_text="world", usage=usage)
    generator.client.responses.create.return_value = api_response
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator._call_response_api(
        inputs=[prompt], long_input_filter_kwargs={"buffer_tokens": 4}
    )

    generator.client.responses.create.assert_called_once_with(
        model="gpt-4o-mini",
        input="hello",
        max_output_tokens=8,
    )
    assert responses == [
        Response(
            input="hello",
            outputs=[{"content": "world", "reasoning_content": None}],
            metadata={
                "trace_id": "abc",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    ]


def test_call_response_api_returns_too_long_error_without_calling_api(
    generator: OpenAIGenerator,
) -> None:
    conversation = Conversation(messages=[Message(role="user", content="x")])
    generator._is_over_context_length = MagicMock(return_value=True)

    responses = generator._call_response_api(
        inputs=[conversation], long_input_filter_kwargs={"buffer_tokens": 1}
    )

    generator.client.responses.create.assert_not_called()
    assert (
        responses[0].outputs[0].content
        == generator.default_too_long_input_error_message
    )


def test_call_response_api_skips_usage_metadata_when_usage_is_none(
    generator: OpenAIGenerator,
) -> None:
    prompt = Prompt(prompt="hello")
    api_response = MagicMock(output_text="world", usage=None)
    generator.client.responses.create.return_value = api_response
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator._call_response_api(inputs=[prompt])

    assert responses[0].metadata is None


def test_chat_forwards_inputs_and_long_input_filter_kwargs(
    generator: OpenAIGenerator,
) -> None:
    conversations = [Conversation(messages=[Message(role="user", content="hi")])]
    spy = MagicMock(return_value=[])
    generator._call_response_api = spy

    generator._chat(
        conversations=conversations,
        long_input_filter_kwargs={"buffer_tokens": 7},
        temperature=0.3,
    )

    spy.assert_called_once_with(
        inputs=conversations,
        long_input_filter_kwargs={"buffer_tokens": 7},
        temperature=0.3,
    )


def test_completion_forwards_inputs_and_long_input_filter_kwargs(
    generator: OpenAIGenerator,
) -> None:
    prompts = [Prompt(prompt="hi")]
    spy = MagicMock(return_value=[])
    generator._call_response_api = spy

    generator._completion(
        prompts=prompts,
        long_input_filter_kwargs={"buffer_tokens": 5},
        temperature=0.1,
    )

    spy.assert_called_once_with(
        inputs=prompts,
        long_input_filter_kwargs={"buffer_tokens": 5},
        temperature=0.1,
    )
