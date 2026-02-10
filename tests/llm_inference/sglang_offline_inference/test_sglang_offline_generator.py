import dataclasses
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


# Test environment may not have optional SGLang dependency installed.
if "sglang" not in sys.modules:
    fake_sglang = types.ModuleType("sglang")
    sys.modules["sglang"] = fake_sglang

if "sglang.srt" not in sys.modules:
    sys.modules["sglang.srt"] = types.ModuleType("sglang.srt")

if "sglang.srt.server_args" not in sys.modules:
    fake_server_args_module = types.ModuleType("sglang.srt.server_args")

    @dataclasses.dataclass
    class _ServerArgs:
        model_path: str
        context_length: int

    fake_server_args_module.ServerArgs = _ServerArgs
    sys.modules["sglang.srt.server_args"] = fake_server_args_module

if "sglang.srt.entrypoints" not in sys.modules:
    sys.modules["sglang.srt.entrypoints"] = types.ModuleType("sglang.srt.entrypoints")

if "sglang.srt.entrypoints.engine" not in sys.modules:
    fake_engine_module = types.ModuleType("sglang.srt.entrypoints.engine")

    class _Engine:
        pass

    fake_engine_module.Engine = _Engine
    sys.modules["sglang.srt.entrypoints.engine"] = fake_engine_module


class DummyTokenizer:
    def __init__(self) -> None:
        self.apply_chat_template_calls: list[dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text)))

    def apply_chat_template(
        self,
        prompt: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        **_kwargs: Any,
    ) -> str | list[int]:
        self.apply_chat_template_calls.append(
            {
                "prompt": prompt,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        if tokenize:
            return list(range(sum(len(m["content"]) for m in prompt)))
        return f"chat:{prompt[0]['content']}"


@dataclasses.dataclass
class FakeServerArgs:
    model_path: str
    context_length: int


class FakeEngine:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.generate_calls: list[dict[str, Any]] = []
        self.tokenizer_manager = types.SimpleNamespace(tokenizer=DummyTokenizer())

    def generate(
        self,
        prompts: list[str],
        *,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": dict(sampling_params),
                "kwargs": kwargs,
            }
        )
        return [{"text": f"response:{prompt}"} for prompt in prompts]


sys.modules["sglang"].Engine = FakeEngine
sys.modules["sglang.srt.server_args"].ServerArgs = FakeServerArgs
sys.modules["sglang.srt.entrypoints.engine"].Engine = FakeEngine

from llm_inference.data import Conversation, Message, Prompt
from llm_inference.sglang_offline_inference import SGLangOfflineGenerator


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> SGLangOfflineGenerator:
    monkeypatch.setattr(
        "llm_inference.sglang_offline_inference.ServerArgs", FakeServerArgs
    )
    monkeypatch.setattr("llm_inference.sglang_offline_inference.Engine", FakeEngine)
    return SGLangOfflineGenerator(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_context_length=32,
        max_output_tokens=8,
        logger=MagicMock(spec=Logger),
    )


def assert_correspondence(
    *,
    responses: list[Any],
    inputs: list[str],
    default_too_long_input_error_message: str,
    has_too_long: bool,
) -> None:
    expected_outputs = (
        expected_outputs_with_middle_too_long(
            inputs=inputs,
            default_too_long_input_error_message=default_too_long_input_error_message,
        )
        if has_too_long
        else [f"response:{input_text}" for input_text in inputs]
    )
    assert len(responses) == len(inputs)
    assert_response_outputs_match_expected_order(
        responses=responses,
        expected_outputs=expected_outputs,
    )


def test_chat_keeps_input_output_correspondence_for_normal_case(
    generator: SGLangOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    conversations = [
        Conversation(messages=[Message(role="user", content=input_text)])
        for input_text in inputs
    ]
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator._chat(conversations=conversations, sampling_params={})

    assert_correspondence(
        responses=responses,
        inputs=inputs,
        default_too_long_input_error_message=generator.default_too_long_input_error_message,
        has_too_long=False,
    )
    assert len(generator.llm.generate_calls) == 1
    assert generator.llm.generate_calls[0]["prompts"] == [
        f"chat:{input_text}" for input_text in inputs
    ]
    assert (
        generator.llm.generate_calls[0]["sampling_params"]["max_tokens"]
        == generator.max_output_tokens
    )
    assert all(
        call["tokenize"] is False and call["add_generation_prompt"] is True
        for call in generator.tokenizer.apply_chat_template_calls
    )


def test_completion_keeps_input_output_correspondence_for_normal_case(
    generator: SGLangOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]
    generator._is_over_context_length = MagicMock(return_value=False)

    responses = generator._completion(prompts=prompts, sampling_params={})

    assert_correspondence(
        responses=responses,
        inputs=inputs,
        default_too_long_input_error_message=generator.default_too_long_input_error_message,
        has_too_long=False,
    )
    assert len(generator.llm.generate_calls) == 1
    assert generator.llm.generate_calls[0]["prompts"] == inputs
    assert (
        generator.llm.generate_calls[0]["sampling_params"]["max_tokens"]
        == generator.max_output_tokens
    )
    assert len(generator.tokenizer.apply_chat_template_calls) == 0


def test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: SGLangOfflineGenerator,
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

    assert_correspondence(
        responses=responses,
        inputs=inputs,
        default_too_long_input_error_message=generator.default_too_long_input_error_message,
        has_too_long=True,
    )
    assert len(generator.llm.generate_calls) == 1
    assert generator.llm.generate_calls[0]["prompts"] == ["chat:a", "chat:c"]


def test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long(
    generator: SGLangOfflineGenerator,
) -> None:
    inputs = ["a", "b", "c"]
    prompts = [Prompt(prompt=input_text) for input_text in inputs]

    def fake_over_context(*, input: Prompt, **_kwargs: Any) -> bool:
        return input.prompt == "b"

    generator._is_over_context_length = MagicMock(side_effect=fake_over_context)

    responses = generator._completion(prompts=prompts, sampling_params={})

    assert_correspondence(
        responses=responses,
        inputs=inputs,
        default_too_long_input_error_message=generator.default_too_long_input_error_message,
        has_too_long=True,
    )
    assert len(generator.llm.generate_calls) == 1
    assert generator.llm.generate_calls[0]["prompts"] == ["a", "c"]


def test_generate_raises_value_error_when_sampling_param_n_is_greater_than_one(
    generator: SGLangOfflineGenerator,
) -> None:
    with pytest.raises(ValueError, match="n > 1 is not supported"):
        generator._generate(prompts=["a"], sampling_params={"n": 2})
