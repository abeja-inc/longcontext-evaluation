import sys
import types
from typing import Any

import pytest


# Test environment may not have optional tokenizer dependencies installed.
if "tiktoken" not in sys.modules:
    fake_tiktoken = types.SimpleNamespace(
        encoding_for_model=lambda *_args, **_kwargs: None,
        get_encoding=lambda *_args, **_kwargs: None,
    )
    sys.modules["tiktoken"] = fake_tiktoken

if "transformers" not in sys.modules:

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
            return None

    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=_AutoTokenizer)

from llm_inference.data import Conversation, Message, Prompt
from llm_inference.token_counter import TokenCounter


class DummyHFTokenizer:
    def __init__(self) -> None:
        self.apply_chat_template_calls: list[dict[str, Any]] = []
        self.encode_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self, prompt: list[dict[str, str]], *, tokenize: bool, **kwargs: Any
    ) -> list[int]:
        self.apply_chat_template_calls.append(
            {"prompt": prompt, "tokenize": tokenize, "kwargs": kwargs}
        )
        return [0, 1, 2, 3]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append(
            {"text": text, "add_special_tokens": add_special_tokens}
        )
        return list(range(len(text)))


class DummyTikTokenizer:
    def __init__(self) -> None:
        self.encode_inputs: list[str] = []

    def encode(self, text: str, disallowed_special: Any = None) -> list[int]:
        self.encode_inputs.append(text)
        return list(range(len(text)))


class WeirdConversation(Conversation):
    @property
    def prompt(self) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": "abc", "name": "alice", "x": 42, "y": None},
            {"role": "assistant", "content": "z"},
        ]


def test_init_huggingface_uses_from_pretrained_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    dummy_hf = DummyHFTokenizer()

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> DummyHFTokenizer:
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return dummy_hf

    monkeypatch.setattr(
        "llm_inference.token_counter.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    counter = TokenCounter(
        tokenizer_name_or_path="hf-model",
        tokenizer_type="huggingface",
    )

    assert counter.tokenizer is dummy_hf
    assert counter.tokenizer_kwargs == {}
    assert captured == {
        "model_name": "hf-model",
        "kwargs": {"trust_remote_code": True},
    }


def test_init_huggingface_forwards_custom_tokenizer_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> DummyHFTokenizer:
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return DummyHFTokenizer()

    monkeypatch.setattr(
        "llm_inference.token_counter.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    _ = TokenCounter(
        tokenizer_name_or_path="hf-model",
        tokenizer_type="huggingface",
        tokenizer_kwargs={"revision": "main", "use_fast": False},
    )

    assert captured == {
        "model_name": "hf-model",
        "kwargs": {
            "trust_remote_code": True,
            "revision": "main",
            "use_fast": False,
        },
    }


def test_init_tiktoken_prefers_encoding_for_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_tik = DummyTikTokenizer()

    def fake_encoding_for_model(model_name: str) -> DummyTikTokenizer:
        assert model_name == "gpt-4o"
        return dummy_tik

    def fail_get_encoding(_: str) -> DummyTikTokenizer:
        raise AssertionError("get_encoding should not be called")

    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.encoding_for_model",
        fake_encoding_for_model,
    )
    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.get_encoding", fail_get_encoding
    )

    counter = TokenCounter(tokenizer_name_or_path="gpt-4o", tokenizer_type="tiktoken")

    assert counter.tokenizer is dummy_tik


def test_init_tiktoken_falls_back_to_get_encoding_on_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_tik = DummyTikTokenizer()

    def fail_encoding_for_model(_: str) -> DummyTikTokenizer:
        raise KeyError("unknown model")

    def fake_get_encoding(model_name: str) -> DummyTikTokenizer:
        assert model_name == "cl100k_base"
        return dummy_tik

    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.encoding_for_model",
        fail_encoding_for_model,
    )
    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.get_encoding", fake_get_encoding
    )

    counter = TokenCounter(
        tokenizer_name_or_path="cl100k_base", tokenizer_type="tiktoken"
    )

    assert counter.tokenizer is dummy_tik


def test_init_raises_for_unsupported_tokenizer_type() -> None:
    with pytest.raises(ValueError, match="Supported tokenizer_type"):
        _ = TokenCounter(
            tokenizer_name_or_path="anything", tokenizer_type="sentencepiece"
        )  # type: ignore[arg-type]


def test_count_tokens_huggingface_conversation_uses_apply_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_hf = DummyHFTokenizer()

    monkeypatch.setattr(
        "llm_inference.token_counter.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: dummy_hf,
    )

    counter = TokenCounter(
        tokenizer_name_or_path="hf-model", tokenizer_type="huggingface"
    )
    conv = Conversation(messages=[Message(role="user", content="hello")])

    token_count = counter.count_tokens(conv, add_generation_prompt=True)

    assert token_count == 4
    assert dummy_hf.apply_chat_template_calls == [
        {
            "prompt": [{"role": "user", "content": "hello"}],
            "tokenize": True,
            "kwargs": {"add_generation_prompt": True},
        }
    ]


def test_count_tokens_huggingface_prompt_and_string_use_encode_without_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_hf = DummyHFTokenizer()

    monkeypatch.setattr(
        "llm_inference.token_counter.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: dummy_hf,
    )

    counter = TokenCounter(
        tokenizer_name_or_path="hf-model", tokenizer_type="huggingface"
    )

    prompt_count = counter.count_tokens(Prompt(prompt="abc"))
    string_count = counter.count_tokens("xy")

    assert prompt_count == 3
    assert string_count == 2
    assert dummy_hf.encode_calls == [
        {"text": "abc", "add_special_tokens": False},
        {"text": "xy", "add_special_tokens": False},
    ]


def test_count_tokens_tiktoken_prompt_and_string_use_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_tik = DummyTikTokenizer()

    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.encoding_for_model",
        lambda _model_name: dummy_tik,
    )

    counter = TokenCounter(tokenizer_name_or_path="gpt-4o", tokenizer_type="tiktoken")

    prompt_count = counter.count_tokens(Prompt(prompt="abc"))
    string_count = counter.count_tokens("wxyz")

    assert prompt_count == 3
    assert string_count == 4
    assert dummy_tik.encode_inputs == ["abc", "wxyz"]


def test_count_tokens_tiktoken_conversation_counts_overhead_and_special_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_tik = DummyTikTokenizer()

    monkeypatch.setattr(
        "llm_inference.token_counter.tiktoken.encoding_for_model",
        lambda _model_name: dummy_tik,
    )

    counter = TokenCounter(tokenizer_name_or_path="gpt-4o", tokenizer_type="tiktoken")
    weird_conversation = WeirdConversation(
        messages=[Message(role="user", content="ignored by overridden prompt")]
    )

    token_count = counter.count_tokens(weird_conversation)

    # message overhead: 3 * 2 messages = 6
    # value lengths encoded: "user"=4, "abc"=3, "alice"=5, "42"=2, "assistant"=9, "z"=1 => 24
    # name field overhead: +1
    # assistant priming tokens: +3
    assert token_count == 34
    assert dummy_tik.encode_inputs == ["user", "abc", "alice", "42", "assistant", "z"]
