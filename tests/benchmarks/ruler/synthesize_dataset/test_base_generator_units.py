from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import Mock


_dummy_token_counter_module = types.ModuleType("llm_inference.token_counter")


class _ImportSafeTokenCounter:
    def __init__(self, tokenizer_name_or_path: str, tokenizer_type: str) -> None:
        del tokenizer_name_or_path, tokenizer_type


_dummy_token_counter_module.TokenCounter = _ImportSafeTokenCounter
sys.modules.setdefault("llm_inference.token_counter", _dummy_token_counter_module)

import pytest
from benchmarks.ruler.synthesize_dataset.base import BaseDatasetGenerator
from benchmarks.ruler.synthesize_dataset.config import BaseSynthesisConfig
from benchmarks.ruler.synthesize_dataset.data_model import BaseDatasetSchema, Content


class _DummyTokenCounter:
    def __init__(self, tokenizer_name_or_path: str, tokenizer_type: str) -> None:
        del tokenizer_name_or_path, tokenizer_type


class _DummyDatasetSchema(BaseDatasetSchema):
    pass


class _DummyGenerator(BaseDatasetGenerator[_DummyDatasetSchema, BaseSynthesisConfig]):
    @property
    def SCHEMA_CLASS(self) -> type[_DummyDatasetSchema]:
        return _DummyDatasetSchema

    def _gen_one_sample(
        self, sample_index: int, num_units: int, **kwargs: Any
    ) -> tuple[Content, dict[str, Any]]:
        del sample_index, kwargs
        return (
            Content(
                user_prompt=f"units:{num_units}",
                answer_prefix="",
                outputs=["ok"],
            ),
            {"target_depth_percent": [0.0]},
        )

    def _prompt_tokens(
        self, user_prompt: str, answer_prefix: str, with_chat_template: bool = True
    ) -> int:
        del answer_prefix, with_chat_template
        num_units = int(user_prompt.split(":", maxsplit=1)[1])
        return num_units * 10


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> _DummyGenerator:
    monkeypatch.setattr(
        "benchmarks.ruler.synthesize_dataset.base.TokenCounter", _DummyTokenCounter
    )
    config = BaseSynthesisConfig(
        save_dirpath=Path("."),
        task="dummy",
        subset="dummy",
        task_prompt_template="{context}",
        answer_prefix_template="A:",
        max_new_tokens=20,
        initial_units=8,
    )
    return _DummyGenerator(config=config, logger=logging.getLogger(__name__))


@pytest.mark.parametrize(
    ("max_context_length", "expected_units", "expected_sampled_units"),
    [
        (100, 8, [8, 16, 12, 9, 8]),
        (95, 7, [8, 4, 6, 7, 8]),
        (370, 35, [8, 16, 32, 64, 48, 39, 35, 37, 36]),
    ],
)
def test_optimal_units_returns_max_valid_unit_and_fixed_search_path(
    generator: _DummyGenerator,
    max_context_length: int,
    expected_units: int,
    expected_sampled_units: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_units: list[int] = []
    original_sample_total_tokens = generator._sample_total_tokens

    def _record_sample_total_tokens(
        num_units: int, **kwargs: Any
    ) -> tuple[int, Content]:
        del kwargs
        sampled_units.append(num_units)
        return original_sample_total_tokens(num_units=num_units)

    monkeypatch.setattr(generator, "_sample_total_tokens", _record_sample_total_tokens)

    actual = generator._optimal_units(max_context_length=max_context_length)

    assert actual == expected_units
    assert sampled_units == expected_sampled_units


def test_optimal_units_returns_one_when_context_not_larger_than_max_new_tokens(
    generator: _DummyGenerator, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_total_tokens_mock = Mock(side_effect=generator._sample_total_tokens)
    monkeypatch.setattr(generator, "_sample_total_tokens", sample_total_tokens_mock)

    actual = generator._optimal_units(
        max_context_length=generator.config.max_new_tokens
    )

    assert actual == 1
    sample_total_tokens_mock.assert_not_called()
