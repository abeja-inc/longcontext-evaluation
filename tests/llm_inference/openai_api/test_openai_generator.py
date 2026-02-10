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

from llm_inference.data import Conversation, Message, Prompt
from llm_inference.openai_api import OpenAIGenerator


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> OpenAIGenerator:
    monkeypatch.setattr(
        "llm_inference.openai_api.tiktoken.encoding_for_model",
        lambda _model_name: DummyTokenizer(),
    )
    return OpenAIGenerator(
        client=MagicMock(),
        model_name="gpt-4o-mini",
        max_context_length=32,
        max_output_tokens=8,
        logger=MagicMock(spec=Logger),
    )


def test_chat_keeps_input_output_correspondence_for_normal_case(
    generator: OpenAIGenerator,
) -> None:
    conversations = [
        Conversation(messages=[Message(role="user", content="a")]),
        Conversation(messages=[Message(role="user", content="b")]),
        Conversation(messages=[Message(role="user", content="c")]),
    ]
    generator._is_over_context_length = MagicMock(return_value=False)

    def fake_create(*, input: list[dict[str, str]], **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input[0]['content']}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator._chat(conversations=conversations)

    assert [r.outputs[0].content for r in responses] == [
        "response:a",
        "response:b",
        "response:c",
    ]


def test_completion_keeps_input_output_correspondence_for_normal_case(
    generator: OpenAIGenerator,
) -> None:
    prompts = [Prompt(prompt="a"), Prompt(prompt="b"), Prompt(prompt="c")]
    generator._is_over_context_length = MagicMock(return_value=False)

    def fake_create(*, input: str, **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator.completion(prompts=prompts)

    assert [r.outputs[0].content for r in responses] == [
        "response:a",
        "response:b",
        "response:c",
    ]


def test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: OpenAIGenerator,
) -> None:
    conversations = [
        Conversation(messages=[Message(role="user", content="a")]),
        Conversation(messages=[Message(role="user", content="b")]),
        Conversation(messages=[Message(role="user", content="c")]),
    ]

    def fake_over_context(*, input: Conversation, **_kwargs: Any) -> bool:
        return input.messages[0].content == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    def fake_create(*, input: list[dict[str, str]], **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input[0]['content']}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator._chat(conversations=conversations)

    assert [r.outputs[0].content for r in responses] == [
        "response:a",
        generator.default_too_long_input_error_message,
        "response:c",
    ]


def test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: OpenAIGenerator,
) -> None:
    prompts = [Prompt(prompt="a"), Prompt(prompt="b"), Prompt(prompt="c")]

    def fake_over_context(*, input: Prompt, **_kwargs: Any) -> bool:
        return input.prompt == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    def fake_create(*, input: str, **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator.completion(prompts=prompts)

    assert [r.outputs[0].content for r in responses] == [
        "response:a",
        generator.default_too_long_input_error_message,
        "response:c",
    ]
