import logging

import pytest

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Message, Prompt, Response


class _TooShortChatGenerator(BaseGenerator):
    def _count_tokens(self, input: Prompt | Conversation, **kwargs: object) -> int:
        return 0

    def _chat(self, *, conversations: list[Conversation], **kwargs: object) -> list[Response]:
        return []

    def _completion(self, *, prompts: list[Prompt], **kwargs: object) -> list[Response]:
        return [
            Response(input=prompt.prompt, outputs=[])
            for prompt in prompts
        ]


class _TooShortCompletionGenerator(BaseGenerator):
    def _count_tokens(self, input: Prompt | Conversation, **kwargs: object) -> int:
        return 0

    def _chat(self, *, conversations: list[Conversation], **kwargs: object) -> list[Response]:
        return [
            Response(input=conversation.prompt, outputs=[])
            for conversation in conversations
        ]

    def _completion(self, *, prompts: list[Prompt], **kwargs: object) -> list[Response]:
        return []


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger(__name__)


def test_chat_raises_runtime_error_when__chat_returns_too_few_responses(
    logger: logging.Logger,
) -> None:
    generator = _TooShortChatGenerator(
        model_name="dummy",
        max_context_length=1,
        max_output_tokens=1,
        logger=logger,
    )
    conversations = [
        Conversation(messages=[Message(role="user", content="hello")]),
        Conversation(messages=[Message(role="user", content="world")]),
    ]

    with pytest.raises(
        RuntimeError,
        match=r"\._chat must return same length as input: in=2 out=0",
    ):
        generator.chat(conversations=conversations)


def test_completion_raises_runtime_error_when__completion_returns_too_few_responses(
    logger: logging.Logger,
) -> None:
    generator = _TooShortCompletionGenerator(
        model_name="dummy",
        max_context_length=1,
        max_output_tokens=1,
        logger=logger,
    )
    prompts = [Prompt(prompt="hello"), Prompt(prompt="world")]

    with pytest.raises(
        RuntimeError,
        match=r"\._completion must return same length as input: in=2 out=0",
    ):
        generator.completion(prompts=prompts)
