import sys
import types
from logging import Logger
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.llm_inference._helpers.response_correspondence import (
    assert_response_outputs_match_expected_order,
    expected_outputs_with_middle_too_long,
)


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

# Test environment may not have optional tokenizer dependencies installed.
if "tiktoken" not in sys.modules:
    sys.modules["tiktoken"] = types.SimpleNamespace(
        encoding_for_model=lambda *_args, **_kwargs: None,
        get_encoding=lambda *_args, **_kwargs: None,
    )

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> "_AutoTokenizer":
            return _AutoTokenizer()

    fake_transformers.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = fake_transformers

from llm_inference.data import Conversation, Message, Prompt
from llm_inference.openai_api_compatible import OpenAICompatibleGenerator


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text)))

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> list[int]:
        return []


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleGenerator:
    monkeypatch.setattr(
        "llm_inference.openai_api.tiktoken.encoding_for_model",
        lambda _model_name: DummyTokenizer(),
    )
    return OpenAICompatibleGenerator(
        client=MagicMock(),
        tokenizer=DummyTokenizer(),
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_context_length=32,
        max_output_tokens=8,
        logger=MagicMock(spec=Logger),
    )


def test_chat_keeps_input_output_correspondence_for_normal_case(
    generator: OpenAICompatibleGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    conversations = [
        Conversation(messages=[Message(role="user", content=input_text)])
        for input_text in inputs
    ]
    generator._is_over_context_length = MagicMock(return_value=False)

    def fake_create(*, input: list[dict[str, str]], **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input[0]['content']}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator._chat(conversations=conversations)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=[f"response:{input_text}" for input_text in inputs],
    )


def test_completion_keeps_input_output_correspondence_for_normal_case(
    generator: OpenAICompatibleGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]
    generator._is_over_context_length = MagicMock(return_value=False)

    def fake_create(*, input: str, **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator.completion(prompts=prompts)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=[f"response:{input_text}" for input_text in inputs],
    )


def test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: OpenAICompatibleGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    conversations = [
        Conversation(messages=[Message(role="user", content=input_text)])
        for input_text in inputs
    ]

    def fake_over_context(*, input: Conversation, **_kwargs: Any) -> bool:
        return input.messages[0].content == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    def fake_create(*, input: list[dict[str, str]], **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input[0]['content']}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator._chat(conversations=conversations)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=expected_outputs_with_middle_too_long(
            inputs=inputs,
            default_too_long_input_error_message=generator.default_too_long_input_error_message,
        ),
    )


def test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: OpenAICompatibleGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]

    def fake_over_context(*, input: Prompt, **_kwargs: Any) -> bool:
        return input.prompt == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    def fake_create(*, input: str, **_kwargs: Any) -> MagicMock:
        return MagicMock(output_text=f"response:{input}", usage=None)

    generator.client.responses.create.side_effect = fake_create

    responses = generator.completion(prompts=prompts)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=expected_outputs_with_middle_too_long(
            inputs=inputs,
            default_too_long_input_error_message=generator.default_too_long_input_error_message,
        ),
    )
