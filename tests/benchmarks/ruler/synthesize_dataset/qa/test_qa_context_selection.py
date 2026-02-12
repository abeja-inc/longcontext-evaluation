import importlib
import logging
import random
import re
import sys
import types
from pathlib import Path
from typing import Any


DOCUMENT_PATTERN = re.compile(
    r"Document \d+:\n(?P<doc>.*?)(?=\n\nDocument \d+:\n|\Z)", re.DOTALL
)


def _install_tokenizer_stubs() -> None:
    tiktoken_module = types.ModuleType("tiktoken")

    class _DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

    def _encoding_for_model(_: str) -> _DummyTokenizer:
        return _DummyTokenizer()

    tiktoken_module.encoding_for_model = _encoding_for_model
    tiktoken_module.get_encoding = _encoding_for_model
    sys.modules.setdefault("tiktoken", tiktoken_module)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules.setdefault("transformers", transformers_module)


def _build_generator(tmp_path: Path) -> Any:
    _install_tokenizer_stubs()
    config_module = importlib.import_module(
        "benchmarks.ruler.synthesize_dataset.config"
    )
    data_model_module = importlib.import_module(
        "benchmarks.ruler.synthesize_dataset.data_model"
    )
    qa_base_module = importlib.import_module(
        "benchmarks.ruler.synthesize_dataset.qa.base"
    )

    class _DummyQAGenerator(qa_base_module.BaseQADatasetGenerator):
        def _process_data(self, raw_data: Any) -> Any:
            raise NotImplementedError

    config = config_module.QASynthesisConfig(
        save_dirpath=tmp_path,
        task="qa",
        subset="unit-test",
        qa_dataset_name="squad",
        qa_dataset_path=tmp_path / "unused.json",
        tokenizer_type="tiktoken",
    )
    generator = _DummyQAGenerator(config=config, logger=logging.getLogger(__name__))
    generator.contexts = [
        "gold context paragraph",
        "same-topic distractor",
        "other-topic distractor",
        "other-topic distractor-2",
    ]
    generator.qas = [
        data_model_module.QAPair(
            question="What is the answer?",
            answers=["answer"],
            context_indices=[0],
            more_context_indices=[1],
        )
    ]
    generator._qa_index_order = [0]
    return generator


def _extract_documents(user_prompt: str) -> list[str]:
    return [match.group("doc") for match in DOCUMENT_PATTERN.finditer(user_prompt)]


def test_gen_one_sample_qa_keeps_gold_context_in_prompt_and_target_context(
    tmp_path: Path,
) -> None:
    generator = _build_generator(tmp_path)
    random.seed(2024)

    content, extra_fields = generator._gen_one_sample(sample_index=0, num_units=3)

    documents = _extract_documents(content.user_prompt)

    assert documents
    assert generator.contexts[0] in documents
    assert extra_fields["target_context"] == generator.contexts[0]


def test_gen_one_sample_qa_mixes_gold_and_distractors_when_available(
    tmp_path: Path,
) -> None:
    generator = _build_generator(tmp_path)
    random.seed(2024)

    content, _ = generator._gen_one_sample(sample_index=0, num_units=4)

    documents = _extract_documents(content.user_prompt)

    assert len(documents) == 4
    assert generator.contexts[0] in documents
    assert any(document != generator.contexts[0] for document in documents)
