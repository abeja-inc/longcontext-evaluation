import logging
from typing import Any

import pytest
from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Prompt, Response


class FixedTokenCountDummyGenerator(BaseGenerator):
    def __init__(
        self,
        *,
        fixed_token_count: int,
        max_context_length: int,
        max_output_tokens: int,
    ) -> None:
        super().__init__(
            model_name="dummy",
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            logger=logging.getLogger(__name__),
        )
        self.fixed_token_count = fixed_token_count

    def _count_tokens(self, input: Prompt | Conversation, **kwargs: object) -> int:
        return self.fixed_token_count

    def _chat(
        self, *, conversations: list[Conversation], **kwargs: object
    ) -> list[Response]:
        return []

    def _completion(self, *, prompts: list[Prompt], **kwargs: object) -> list[Response]:
        return []


@pytest.mark.parametrize(
    "input_data",
    [
        Prompt(prompt="hello"),
        Conversation(messages=[{"role": "user", "content": "hello"}]),
    ],
)
def test_is_over_context_length_allows_equal_boundary(
    input_data: Prompt | Conversation,
) -> None:
    max_context_length = 100
    max_output_tokens = 20
    buffer_tokens = 10
    input_tokens = max_context_length - max_output_tokens - buffer_tokens

    generator = FixedTokenCountDummyGenerator(
        fixed_token_count=input_tokens,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
    )

    result = generator._is_over_context_length(
        input=input_data,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
        buffer_tokens=buffer_tokens,
    )

    assert result is False


@pytest.mark.parametrize(
    "input_data",
    [
        Prompt(prompt="hello"),
        Conversation(messages=[{"role": "user", "content": "hello"}]),
    ],
)
def test_is_over_context_length_returns_true_when_total_exceeds_by_one(
    input_data: Prompt | Conversation,
) -> None:
    max_context_length = 100
    max_output_tokens = 20
    buffer_tokens = 10
    input_tokens = max_context_length - max_output_tokens - buffer_tokens + 1

    generator = FixedTokenCountDummyGenerator(
        fixed_token_count=input_tokens,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
    )

    result = generator._is_over_context_length(
        input=input_data,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
        buffer_tokens=buffer_tokens,
    )

    assert result is True


class StringLengthDummyGenerator(BaseGenerator):
    def _count_tokens(self, input: str, **kwargs: Any) -> int:
        return len(input)

    def _chat(self, *, conversations: list[Any], **kwargs: Any) -> list[Any]:
        return []

    def _completion(self, *, prompts: list[Any], **kwargs: Any) -> list[Any]:
        return []


def test_filter_long_inputs_keeps_order_and_returns_original_skip_indices() -> None:
    generator = StringLengthDummyGenerator(
        model_name="dummy",
        max_context_length=8,
        max_output_tokens=2,
        logger=logging.getLogger(__name__),
    )
    inputs = ["ok", "too-long", "fit", "overflow"]

    filtered_inputs, skip_idx = generator._filter_long_inputs(
        inputs=inputs,
        max_context_length=8,
        max_output_tokens=2,
    )

    assert filtered_inputs == ["ok", "fit"]
    assert skip_idx == [1, 3]
    assert len(filtered_inputs) + len(skip_idx) == len(inputs)
