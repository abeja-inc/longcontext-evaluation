from logging import getLogger
from typing import Any

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Message, Prompt


class LoggingDummyGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__(
            model_name="dummy-model",
            max_context_length=1024,
            max_output_tokens=128,
            logger=getLogger(__name__),
        )
        self.count_tokens_kwargs_history: list[dict[str, Any]] = []
        self.count_tokens_call_count = 0

    def _count_tokens(self, input: Prompt | Conversation, **kwargs: Any) -> int:
        self.count_tokens_kwargs_history.append(kwargs)
        self.count_tokens_call_count += 1
        return 1

    def _chat(
        self, *, conversations: list[Conversation], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"text": "ok"} for _ in conversations]

    def _completion(self, *, prompts: list[Prompt], **kwargs: Any) -> list[dict[str, Any]]:
        return [{"text": "ok"} for _ in prompts]


def test_is_over_context_length_passes_kwargs_to_count_tokens() -> None:
    generator = LoggingDummyGenerator()

    _ = generator._is_over_context_length(
        input=Prompt(prompt="hello"),
        max_context_length=10,
        max_output_tokens=2,
        buffer_tokens=0,
        foo="bar",
    )

    assert generator.count_tokens_call_count == 1
    assert generator.count_tokens_kwargs_history == [{"foo": "bar"}]


def test_filter_long_inputs_passes_kwargs_to_each_count_tokens_call() -> None:
    generator = LoggingDummyGenerator()
    inputs = [
        Prompt(prompt="hello"),
        Prompt(prompt="world"),
        Conversation(messages=[Message(role="user", content="!")]),
    ]

    _ = generator._filter_long_inputs(
        inputs=inputs,
        max_context_length=10,
        max_output_tokens=2,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert generator.count_tokens_call_count == len(inputs)
    assert generator.count_tokens_kwargs_history == [
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ]
