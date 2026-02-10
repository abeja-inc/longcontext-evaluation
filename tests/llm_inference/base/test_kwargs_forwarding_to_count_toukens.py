from logging import getLogger
from typing import Any

from llm_inference.base import BaseGenerator


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

    def _count_tokens(self, input: str | list[dict[str, str]], **kwargs: Any) -> int:
        self.count_tokens_kwargs_history.append(kwargs)
        self.count_tokens_call_count += 1
        return 1

    def _chat(
        self, *, conversations: list[list[dict[str, str]]], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"text": "ok"} for _ in conversations]

    def _completion(self, *, prompts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        return [{"text": "ok"} for _ in prompts]


def test_is_over_context_length_passes_kwargs_to_count_tokens() -> None:
    generator = LoggingDummyGenerator()

    _ = generator._is_over_context_length(
        input="hello",
        max_context_length=10,
        max_output_tokens=2,
        buffer_tokens=0,
        foo="bar",
    )

    assert generator.count_tokens_call_count == 1
    assert generator.count_tokens_kwargs_history == [{"foo": "bar"}]


def test_filter_long_inputs_passes_kwargs_to_each_count_tokens_call() -> None:
    generator = LoggingDummyGenerator()
    prompts = ["hello", "world", "!"]

    _ = generator._filter_long_inputs(
        inputs=prompts,
        max_context_length=10,
        max_output_tokens=2,
        chat_template_kwargs={"enable_thinking": True},
    )

    assert generator.count_tokens_call_count == len(prompts)
    assert generator.count_tokens_kwargs_history == [
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ]
