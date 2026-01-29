import json
import logging
import random
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "packages" / "benchmarks" / "src"))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    ),
)

from benchmarks.ruler.synthesize_dataset.config import QASynthesisConfig
from benchmarks.ruler.synthesize_dataset.data_model import ProcessedQAData, QAPair
from benchmarks.ruler.synthesize_dataset.qa.base import BaseQADatasetGenerator


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        return messages[0]["content"]

    def encode(self, text):
        return list(text.encode("utf-8"))


class DummyQAGenerator(BaseQADatasetGenerator):
    def _process_data(self, raw_data):
        contexts = [item["context"] for item in raw_data]
        unique_contexts = list(dict.fromkeys(contexts))
        idx_contexts = {context: idx for idx, context in enumerate(unique_contexts)}
        qas = [
            QAPair(
                question=item["question"],
                answers=[item["answer"]],
                context_indices=[idx_contexts[item["context"]]],
                more_context_indices=None,
            )
            for item in raw_data
        ]
        return ProcessedQAData(contexts=unique_contexts, qas=qas)


@pytest.fixture
def qa_dataset_path(tmp_path):
    data = [
        {"question": "q0", "answer": "a0", "context": "c0"},
        {"question": "q1", "answer": "a1", "context": "c1"},
        {"question": "q2", "answer": "a2", "context": "c2"},
        {"question": "q3", "answer": "a3", "context": "c3"},
        {"question": "q4", "answer": "a4", "context": "c4"},
    ]
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def qa_config(tmp_path, qa_dataset_path):
    return QASynthesisConfig(
        save_dirpath=tmp_path,
        task="qa",
        subset="unit",
        qa_dataset_name="squad",
        qa_dataset_path=qa_dataset_path,
        task_prompt_template="Q: {query}\n{context}",
        answer_prefix_template="A:",
        document_prompt="Doc {i}: {document}",
        context_lengths=[8],
        num_samples=3,
        random_seed=123,
        max_new_tokens=1,
    )


@pytest.fixture
def generator(monkeypatch, qa_config):
    from benchmarks.ruler.synthesize_dataset import base as base_module

    monkeypatch.setattr(
        base_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    return DummyQAGenerator(config=qa_config, logger=logging.getLogger(__name__))


def test_qa_index_order_is_shuffled(generator):
    generator.prepare()

    expected_order = list(range(len(generator.qas)))
    rng = random.Random(generator.config.random_seed)
    rng.shuffle(expected_order)

    assert generator._qa_index_order == expected_order


def test_gen_one_sample_uses_shuffled_order(generator):
    generator.prepare()

    expected_order = list(range(len(generator.qas)))
    rng = random.Random(generator.config.random_seed)
    rng.shuffle(expected_order)

    _, extra_fields = generator._gen_one_sample(sample_index=0, num_units=2)

    assert extra_fields["question"] == generator.qas[expected_order[0]].question
