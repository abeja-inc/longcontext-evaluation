from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

from llm_inference.data import Conversation, Message, Prompt


def _install_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_tiktoken = types.ModuleType("tiktoken")

    class DummyEncoding:
        def encode(self, text: str):
            return list(range(len(str(text))))

    dummy_tiktoken.encoding_for_model = lambda name: DummyEncoding()
    dummy_tiktoken.get_encoding = lambda name: DummyEncoding()
    monkeypatch.setitem(sys.modules, "tiktoken", dummy_tiktoken)


def _install_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_module = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - just for import compatibility
        pass

    openai_module.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    types_module = types.ModuleType("openai.types")
    responses_module = types.ModuleType("openai.types.responses")

    class Response:  # pragma: no cover - used for type hints
        pass

    responses_module.Response = Response
    monkeypatch.setitem(sys.modules, "openai.types", types_module)
    monkeypatch.setitem(sys.modules, "openai.types.responses", responses_module)


def _install_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = types.ModuleType("transformers")

    class DummyAutoTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return list(range(len(str(text))))

        def apply_chat_template(self, prompt, tokenize: bool = True, **kwargs):
            return list(range(len(prompt)))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _install_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_module = types.ModuleType("vllm")
    outputs_module = types.ModuleType("vllm.outputs")

    class DummyOutput:
        def __init__(self, text: str):
            self.text = text

    class RequestOutput(list):
        def __init__(self, outputs):
            super().__init__()
            self.outputs = outputs

    class SamplingParams:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs.get("max_tokens")

        def __repr__(self) -> str:
            return f"SamplingParams(max_tokens={self.max_tokens})"

    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return list(range(len(str(text))))

        def apply_chat_template(self, prompt, tokenize: bool = True, **kwargs):
            return list(range(len(prompt)))

    class LLM:
        def __init__(self, *args, **kwargs):
            self._tokenizer = DummyTokenizer()

        def get_tokenizer(self):
            return self._tokenizer

        def chat(self, prompts, sampling_params, chat_template_kwargs=None, **kwargs):
            return [RequestOutput([DummyOutput("reason|answer")]) for _ in prompts]

        def generate(self, prompts, sampling_params, **kwargs):
            return [RequestOutput([DummyOutput("reason|answer")]) for _ in prompts]

    vllm_module.LLM = LLM
    vllm_module.SamplingParams = SamplingParams
    outputs_module.RequestOutput = RequestOutput
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.outputs", outputs_module)


def _install_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    server_args_module = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        def __init__(self, **kwargs):
            self.model_path = kwargs.get("model_path")
            self.context_length = kwargs.get("context_length")

    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return list(range(len(str(text))))

        def apply_chat_template(self, prompt, **kwargs):
            return "|".join([m["content"] for m in prompt])

    class TokenizerManager:
        def __init__(self):
            self.tokenizer = DummyTokenizer()

    class Engine:
        def __init__(self, **kwargs):
            self.tokenizer_manager = TokenizerManager()

        def generate(self, prompts, sampling_params=None, **kwargs):
            return [{"text": "reason|answer"} for _ in prompts]

    sglang_module.Engine = Engine
    server_args_module.ServerArgs = ServerArgs
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)


class DummyReasoningParser:
    def parse(self, text: str) -> tuple[str, str]:
        reasoning, _, content = text.partition("|")
        return reasoning, content


def test_openai_generator_long_input_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiktoken(monkeypatch)
    _install_openai(monkeypatch)
    module = importlib.reload(importlib.import_module("llm_inference.openai_api"))

    class DummyUsage:
        def model_dump(self):
            return {"total_tokens": 1}

    class DummyResponse:
        def __init__(self, text: str):
            self.output_text = text
            self.usage = DummyUsage()

    class DummyResponses:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return DummyResponse("ok")

    class DummyClient:
        def __init__(self):
            self.responses = DummyResponses()

    generator = module.OpenAIGenerator(
        client=DummyClient(),
        model_name="dummy",
        max_context_length=5,
        max_output_tokens=2,
        logger=logging.getLogger("test"),
    )

    prompts = [Prompt(prompt="long"), Prompt(prompt="ok")]
    responses = generator.completion(prompts=prompts)

    assert len(responses) == 2
    assert responses[0].outputs[0].content == generator.default_too_long_input_error_message
    assert responses[1].outputs[0].content == "ok"


def test_openai_compatible_generator_long_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiktoken(monkeypatch)
    _install_openai(monkeypatch)
    _install_transformers(monkeypatch)
    module = importlib.reload(importlib.import_module("llm_inference.openai_api_compatible"))

    class DummyResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text="ok", usage=None)

    class DummyClient:
        def __init__(self):
            self.responses = DummyResponses()

    tokenizer = sys.modules["transformers"].AutoTokenizer.from_pretrained("dummy")
    generator = module.OpenAICompatibleGenerator(
        client=DummyClient(),
        tokenizer=tokenizer,
        model_name="dummy",
        max_context_length=15,
        max_output_tokens=2,
        logger=logging.getLogger("test"),
    )

    conversations = [
        Conversation(messages=[Message(role="user", content="toolong")]),
        Conversation(messages=[Message(role="user", content="ok")]),
    ]
    responses = generator.chat(conversations=conversations)
    assert len(responses) == 2
    assert responses[0].outputs[0].content == generator.default_too_long_input_error_message
    assert responses[1].outputs[0].content == "ok"


def test_vllm_offline_generator_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_vllm(monkeypatch)
    module = importlib.reload(importlib.import_module("llm_inference.vllm_offline_inference"))

    generator = module.VLLMOfflineGenerator(
        model_name="dummy",
        max_context_length=5,
        max_output_tokens=2,
        logger=logging.getLogger("test"),
        reasoning_parser=DummyReasoningParser(),
    )

    conversations = [
        Conversation(messages=[Message(role="user", content="short")]),
        Conversation(
            messages=[
                Message(role="user", content="m1"),
                Message(role="assistant", content="m2"),
                Message(role="user", content="m3"),
                Message(role="assistant", content="m4"),
            ]
        ),
    ]
    responses = generator.chat(conversations=conversations, sampling_params={})

    assert len(responses) == 2
    assert responses[0].outputs[0].reasoning_content == "reason"
    assert responses[0].outputs[0].content == "answer"
    assert responses[1].outputs[0].content == generator.default_too_long_input_error_message


def test_vllm_offline_generator_no_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_vllm(monkeypatch)
    module = importlib.reload(importlib.import_module("llm_inference.vllm_offline_inference"))

    generator = module.VLLMOfflineGenerator(
        model_name="dummy",
        max_context_length=10,
        max_output_tokens=2,
        logger=logging.getLogger("test"),
        reasoning_parser=None,
    )

    conversations = [Conversation(messages=[Message(role="user", content="short")])]
    responses = generator.chat(conversations=conversations, sampling_params={})

    assert responses[0].outputs[0].reasoning_content is None


def test_sglang_offline_generator_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sglang(monkeypatch)
    module = importlib.reload(importlib.import_module("llm_inference.sglang_offline_inference"))

    generator = module.SGLangOfflineGenerator(
        model_name="dummy",
        max_context_length=6,
        max_output_tokens=2,
        logger=logging.getLogger("test"),
        reasoning_parser=DummyReasoningParser(),
    )

    prompts = [Prompt(prompt="ok"), Prompt(prompt="toolong")]
    responses = generator.completion(prompts=prompts, sampling_params={})

    assert len(responses) == 2
    assert responses[0].outputs[0].reasoning_content == "reason"
    assert responses[0].outputs[0].content == "answer"
    assert responses[1].outputs[0].content == generator.default_too_long_input_error_message
