from __future__ import annotations

import logging

import pytest

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Message, Prompt, Response


class _DummyGenerator(BaseGenerator):
    def _count_tokens(self, input: Prompt | Conversation, **kwargs: object) -> int:
        return 0

    def _chat(self, *, conversations: list[Conversation], **kwargs: object) -> list[Response]:
        return []

    def _completion(self, *, prompts: list[Prompt], **kwargs: object) -> list[Response]:
        return []


def test_chat_length_mismatch_raises() -> None:
    generator = _DummyGenerator(
        model_name="dummy",
        max_context_length=10,
        max_output_tokens=1,
        logger=logging.getLogger("test"),
    )
    conversations = [Conversation(messages=[Message(role="user", content="hi")])]
    with pytest.raises(RuntimeError, match=r"_chat must return same length as input"):
        generator.chat(conversations=conversations)


def test_completion_length_mismatch_raises() -> None:
    generator = _DummyGenerator(
        model_name="dummy",
        max_context_length=10,
        max_output_tokens=1,
        logger=logging.getLogger("test"),
    )
    prompts = [Prompt(prompt="hi"), Prompt(prompt="bye")]
    with pytest.raises(RuntimeError, match=r"_completion must return same length as input"):
        generator.completion(prompts=prompts)
