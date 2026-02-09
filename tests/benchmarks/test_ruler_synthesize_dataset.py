from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks.ruler.synthesize_dataset.base import BaseDatasetGenerator
from benchmarks.ruler.synthesize_dataset.config import BaseSynthesisConfig, NIAHSynthesisConfig, QASynthesisConfig
from benchmarks.ruler.synthesize_dataset.data_model import BaseDatasetSchema, Content, ProcessedQAData, QAPair
from benchmarks.ruler.synthesize_dataset.qa.base import BaseQADatasetGenerator


class DummyTokenCounter:
    def count_tokens(self, input, **kwargs: Any) -> int:
        if hasattr(input, "prompt"):
            prompt = input.prompt
            if isinstance(prompt, list):
                return sum(len(m.get("content", "")) for m in prompt)
            return len(str(prompt))
        return len(str(input))


class DummyDatasetGenerator(BaseDatasetGenerator[BaseDatasetSchema, BaseSynthesisConfig]):
    @property
    def SCHEMA_CLASS(self):
        return BaseDatasetSchema

    def __init__(self, config: BaseSynthesisConfig, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self._attempts = 0

    def _optimal_units(self, max_context_length: int, **kwargs: Any) -> int:
        return 1

    def _gen_one_sample(self, sample_index: int, num_units: int, **kwargs: Any):
        self._attempts += 1
        if self._attempts == 1:
            user_prompt = "x" * 50
        else:
            user_prompt = "short prompt"
        content = Content(user_prompt=user_prompt, answer_prefix="ANSWER", outputs=["ok"])
        extra_fields = {"target_depth_percent": [42.0]}
        return content, extra_fields


class DummyQAGenerator(BaseQADatasetGenerator):
    def _process_data(self, raw_data: Any) -> ProcessedQAData:
        return ProcessedQAData(
            contexts=["Doc A", "Doc B"],
            qas=[
                QAPair(
                    question="What?",
                    answers=["Answer"],
                    context_indices=[0],
                    more_context_indices=[1],
                )
            ],
        )


def test_base_dataset_generator_retry_and_depth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "benchmarks.ruler.synthesize_dataset.base.TokenCounter", DummyTokenCounter
    )
    config = BaseSynthesisConfig(
        save_dirpath=tmp_path,
        task="task",
        subset="subset",
        task_prompt_template="prompt",
        answer_prefix_template="ANSWER",
        context_lengths=[20],
        num_samples=1,
        max_new_tokens=1,
        min_units=1,
        initial_units=1,
        retry_limit=2,
    )
    generator = DummyDatasetGenerator(config, logging.getLogger("test"))
    samples = generator._generate_samples(max_context_length=20)

    assert len(samples) == 1
    assert samples[0].target_depth_percent == [42.0]
    assert generator._attempts >= 2
    assert samples[0].content.answer_prefix == "ANSWER"


def test_niah_generator_depth_and_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "benchmarks.ruler.synthesize_dataset.base.TokenCounter", DummyTokenCounter
    )
    nltk_module = types.ModuleType("nltk")
    nltk_module.download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.tokenize", types.SimpleNamespace(sent_tokenize=lambda text: [text]))
    wonderwords_module = types.ModuleType("wonderwords")
    wonderwords_module.random_word = types.SimpleNamespace(
        _get_words_from_text_file=lambda name: ["blue", "dog"]
    )
    monkeypatch.setitem(sys.modules, "wonderwords", wonderwords_module)

    niah_module = importlib.reload(
        importlib.import_module("benchmarks.ruler.synthesize_dataset.niah.niah")
    )
    config = NIAHSynthesisConfig(
        save_dirpath=tmp_path,
        task="niah",
        subset="test",
        paulgraham_essay_path=tmp_path / "essay.json",
        type_haystack="noise",
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
    )
    generator = niah_module.NIAHDatasetGenerator(config, logging.getLogger("test"))
    generator.haystack_source = ["haystack sentence"]
    generator.words = ["blue-dog"]

    content, extra_fields = generator._gen_one_sample(sample_index=0, num_units=3)
    assert content.user_prompt
    assert content.answer_prefix in content.user_prompt
    assert extra_fields["target_depth_percent"]
    assert all(0.0 <= depth <= 100.0 for depth in extra_fields["target_depth_percent"])


def test_base_qa_generator_depth_and_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "benchmarks.ruler.synthesize_dataset.base.TokenCounter", DummyTokenCounter
    )
    data_path = tmp_path / "qa.json"
    data_path.write_text(json.dumps([{"dummy": True}]), encoding="utf-8")

    config = QASynthesisConfig(
        save_dirpath=tmp_path,
        task="qa",
        subset="test",
        qa_dataset_name="squad",
        qa_dataset_path=data_path,
        num_samples=1,
        context_lengths=[50],
    )
    generator = DummyQAGenerator(config, logging.getLogger("test"))
    generator.prepare()

    content, extra_fields = generator._gen_one_sample(sample_index=0, num_units=2)
    assert "Document 1:" in content.user_prompt
    assert content.answer_prefix == config.answer_prefix_template
    assert extra_fields["target_depth_percent"]
