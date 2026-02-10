import logging
from typing import Any

from llm_inference.base import BaseGenerator


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
