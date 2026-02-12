import importlib
import logging
import re
import sys
import types

import pytest


if "tiktoken" not in sys.modules:
    tiktoken_stub = types.ModuleType("tiktoken")

    class _DummyEncoding:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    def _encoding_for_model(_: str) -> _DummyEncoding:
        raise KeyError

    def _get_encoding(_: str) -> _DummyEncoding:
        return _DummyEncoding()

    tiktoken_stub.encoding_for_model = _encoding_for_model
    tiktoken_stub.get_encoding = _get_encoding
    sys.modules["tiktoken"] = tiktoken_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise RuntimeError("AutoTokenizer should not be used in this test")

    transformers_stub.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = transformers_stub


_qa_config_module = importlib.import_module(
    "benchmarks.ruler.synthesize_dataset.config"
)
_qa_data_module = importlib.import_module(
    "benchmarks.ruler.synthesize_dataset.data_model"
)
_qa_base_module = importlib.import_module("benchmarks.ruler.synthesize_dataset.qa.base")

QASynthesisConfig = _qa_config_module.QASynthesisConfig
ProcessedQAData = _qa_data_module.ProcessedQAData
QAPair = _qa_data_module.QAPair
BaseQADatasetGenerator = _qa_base_module.BaseQADatasetGenerator


class FixedQAGenerator(BaseQADatasetGenerator):
    def _prepare(self, **kwargs: object) -> None:
        self.contexts = [
            "Context A.",
            "Context B.",
            "Context C.",
            "Context D.",
        ]
        self.qas = [
            QAPair(
                question="Which contexts are gold?",
                answers=["Context B and Context D"],
                context_indices=[1, 3],
                more_context_indices=[2],
            )
        ]
        self._qa_index_order = [0]

    def _process_data(self, raw_data: object) -> ProcessedQAData:
        raise NotImplementedError


@pytest.fixture
def generator(tmp_path):
    config = QASynthesisConfig(
        save_dirpath=tmp_path,
        task="qa",
        subset="fixed",
        tokenizer_name_or_path="cl100k_base",
        tokenizer_type="tiktoken",
        qa_dataset_name="squad",
        qa_dataset_path=tmp_path / "dummy.json",
        task_prompt_template="{context}\n\nQuestion: {query}",
        answer_prefix_template="Answer:",
        document_prompt="Document {i}:\n{document}",
    )
    test_generator = FixedQAGenerator(config=config, logger=logging.getLogger(__name__))
    test_generator.prepare()
    return test_generator


def _extract_documents(user_prompt: str) -> list[tuple[int, str]]:
    pattern = r"Document (\d+):\n(.*?)(?=\n\nDocument \d+:\n|\n\nQuestion:|\Z)"
    return [
        (int(number), body) for number, body in re.findall(pattern, user_prompt, re.S)
    ]


@pytest.mark.parametrize("num_units", [1, 2])
def test_gen_one_sample_includes_gold_docs_and_answer_candidates(generator, num_units):
    content, extra_fields = generator._gen_one_sample(
        sample_index=0, num_units=num_units
    )

    for idx in generator.qas[0].context_indices:
        assert generator.contexts[idx] in extra_fields["target_context"]

    assert content.outputs == generator.qas[0].answers

    depth_percent = extra_fields["target_depth_percent"][0]
    assert depth_percent == -1.0 or 0.0 <= depth_percent <= 100.0


@pytest.mark.parametrize(
    ("num_units", "expected_document_count"),
    [
        (1, 2),  # num_units < gold文書数
        (6, 6),  # num_units > 全context数
    ],
)
def test_gen_one_sample_boundary_cases_for_document_count_and_prompt_order(
    generator, num_units, expected_document_count
):
    content, _ = generator._gen_one_sample(sample_index=0, num_units=num_units)

    documents = _extract_documents(content.user_prompt)
    assert len(documents) == expected_document_count

    document_numbers = [number for number, _ in documents]
    assert document_numbers == list(range(1, len(documents) + 1))

    for number, body in documents:
        expected_block = generator.config.document_prompt.format(
            i=number, document=body
        )
        assert expected_block in content.user_prompt
