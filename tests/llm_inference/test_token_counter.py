from __future__ import annotations

import importlib
import sys
import types

import pytest

from llm_inference.data import Conversation, Message, Prompt


def _load_token_counter(monkeypatch: pytest.MonkeyPatch):
    dummy_tiktoken = types.ModuleType("tiktoken")

    class DummyEncoding:
        def __init__(self, name: str):
            self.name = name

        def encode(self, text: str):
            return list(range(len(str(text))))

    def encoding_for_model(name: str):
        if name == "missing-model":
            raise KeyError("missing")
        return DummyEncoding(name)

    def get_encoding(name: str):
        return DummyEncoding(name)

    dummy_tiktoken.encoding_for_model = encoding_for_model
    dummy_tiktoken.get_encoding = get_encoding
    monkeypatch.setitem(sys.modules, "tiktoken", dummy_tiktoken)

    transformers = types.ModuleType("transformers")

    class DummyAutoTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return list(range(len(str(text)) + 1))

        def apply_chat_template(self, prompt, tokenize: bool = True, **kwargs):
            return list(range(len(prompt) + 2))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    module = importlib.import_module("llm_inference.token_counter")
    return importlib.reload(module)


def test_token_counter_huggingface_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    token_counter = _load_token_counter(monkeypatch)
    counter = token_counter.TokenCounter(
        tokenizer_name_or_path="dummy",
        tokenizer_type="huggingface",
    )

    assert counter.count_tokens(Prompt(prompt="hi")) == 3
    assert counter.count_tokens("ok") == 3

    conversation = Conversation(
        messages=[Message(role="user", content="hi"), Message(role="assistant", content="yo")]
    )
    assert counter.count_tokens(conversation) == 4


def test_token_counter_tiktoken_branches_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    token_counter = _load_token_counter(monkeypatch)
    counter = token_counter.TokenCounter(
        tokenizer_name_or_path="missing-model",
        tokenizer_type="tiktoken",
    )

    conversation = Conversation(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="yo"),
        ]
    )
    prompt = Prompt(prompt="hello")

    assert counter.count_tokens(prompt) == 5
    assert counter.count_tokens("abc") == 3
    assert counter.count_tokens(conversation) == 26


def test_token_counter_invalid_type(monkeypatch: pytest.MonkeyPatch) -> None:
    token_counter = _load_token_counter(monkeypatch)
    with pytest.raises(ValueError, match="Supported tokenizer_type"):
        token_counter.TokenCounter(
            tokenizer_name_or_path="dummy",
            tokenizer_type="unknown",  # type: ignore[arg-type]
        )
