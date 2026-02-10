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


# Test environment may not have optional vLLM dependency installed.
if "vllm" not in sys.modules:

    class _SamplingParams:
        def __init__(self, max_tokens: int | None = None, **_kwargs: Any) -> None:
            self.max_tokens = max_tokens

    class _LLM:
        pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    fake_vllm.LLM = _LLM
    sys.modules["vllm"] = fake_vllm

if "vllm.outputs" not in sys.modules:
    fake_vllm_outputs = types.ModuleType("vllm.outputs")

    class _RequestOutput:
        pass

    fake_vllm_outputs.RequestOutput = _RequestOutput
    sys.modules["vllm.outputs"] = fake_vllm_outputs

from llm_inference.data import Conversation, Message, Prompt
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text)))

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> list[int]:
        return []


class FakeLLM:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self._tokenizer = DummyTokenizer()

    def get_tokenizer(self) -> DummyTokenizer:
        return self._tokenizer

    def chat(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        sampling_params: Any,
        chat_template_kwargs: dict[str, Any],
        **_kwargs: Any,
    ) -> list[Any]:
        self.chat_calls.append(
            {
                "conversations": conversations,
                "sampling_params": sampling_params,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(text=f"response:{conversation[0]['content']}")
                ]
            )
            for conversation in conversations
        ]

    def generate(
        self,
        prompts: list[str],
        *,
        sampling_params: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
            }
        )
        return [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"response:{p}")])
            for p in prompts
        ]


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> VLLMOfflineGenerator:
    monkeypatch.setattr("llm_inference.vllm_offline_inference.LLM", FakeLLM)
    return VLLMOfflineGenerator(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_context_length=32,
        max_output_tokens=8,
        logger=MagicMock(spec=Logger),
    )


def test_chat_keeps_input_output_correspondence_for_normal_case(
    generator: VLLMOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    conversations = [
        Conversation(messages=[Message(role="user", content=input_text)])
        for input_text in inputs
    ]
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator._chat(conversations=conversations, sampling_params={})

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=[f"response:{input_text}" for input_text in inputs],
    )
    assert len(responses) == len(conversations)
    assert len(generator.llm.chat_calls) == 1
    assert len(generator.llm.generate_calls) == 0
    assert (
        generator.llm.chat_calls[0]["sampling_params"].max_tokens
        == generator.max_output_tokens
    )


def test_completion_keeps_input_output_correspondence_for_normal_case(
    generator: VLLMOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator.completion(prompts=prompts)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=[f"response:{input_text}" for input_text in inputs],
    )
    assert len(responses) == len(prompts)
    assert len(generator.llm.generate_calls) == 1
    assert len(generator.llm.chat_calls) == 0
    assert (
        generator.llm.generate_calls[0]["sampling_params"].max_tokens
        == generator.max_output_tokens
    )


def test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: VLLMOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    conversations = [
        Conversation(messages=[Message(role="user", content=input_text)])
        for input_text in inputs
    ]

    def fake_over_context(*, input: Conversation, **_kwargs: Any) -> bool:
        return input.messages[0].content == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    responses = generator._chat(conversations=conversations, sampling_params={})

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=expected_outputs_with_middle_too_long(
            inputs=inputs,
            default_too_long_input_error_message=generator.default_too_long_input_error_message,
        ),
    )
    assert len(responses) == len(conversations)
    assert len(generator.llm.chat_calls) == 1
    assert len(generator.llm.generate_calls) == 0
    assert (
        generator.llm.chat_calls[0]["sampling_params"].max_tokens
        == generator.max_output_tokens
    )


def test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: VLLMOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]

    def fake_over_context(*, input: Prompt, **_kwargs: Any) -> bool:
        return input.prompt == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    responses = generator.completion(prompts=prompts)

    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=expected_outputs_with_middle_too_long(
            inputs=inputs,
            default_too_long_input_error_message=generator.default_too_long_input_error_message,
        ),
    )
    assert len(responses) == len(prompts)
    assert len(generator.llm.generate_calls) == 1
    assert len(generator.llm.chat_calls) == 0
    assert (
        generator.llm.generate_calls[0]["sampling_params"].max_tokens
        == generator.max_output_tokens
    )
