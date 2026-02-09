from pathlib import Path
import logging
import sys
import types

TESTS_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_SRC = TESTS_ROOT / "packages" / "benchmarks" / "src"
LLM_INFERENCE_SRC = TESTS_ROOT / "packages" / "llm_inference" / "src"
sys.path.append(str(BENCHMARKS_SRC))
sys.path.append(str(LLM_INFERENCE_SRC))

benchmarks_module = types.ModuleType("benchmarks")
benchmarks_module.__path__ = [str(BENCHMARKS_SRC / "benchmarks")]
sys.modules.setdefault("benchmarks", benchmarks_module)

longbench_v2_module = types.ModuleType("benchmarks.longbench_v2")
longbench_v2_module.__path__ = [
    str(BENCHMARKS_SRC / "benchmarks" / "longbench_v2")
]
sys.modules.setdefault("benchmarks.longbench_v2", longbench_v2_module)

predict_module = types.ModuleType("benchmarks.longbench_v2.predict")
predict_module.__path__ = [
    str(BENCHMARKS_SRC / "benchmarks" / "longbench_v2" / "predict")
]
sys.modules.setdefault("benchmarks.longbench_v2.predict", predict_module)

predict_data_module = types.ModuleType("benchmarks.longbench_v2.predict.data")


class LongBenchV2Output:
    def __init__(self, *, output: str, answer: str) -> None:
        self.output = output
        self.answer = answer


predict_data_module.LongBenchV2Output = LongBenchV2Output
sys.modules.setdefault("benchmarks.longbench_v2.predict.data", predict_data_module)

import pytest

from benchmarks.config import SubtaskConfig
from benchmarks.longbench_v2.evaluate.metrics import LongBenchV2Metrics
from benchmarks.longbench_v2.predict.data import LongBenchV2Output
from benchmarks.longbench_v2.settings import LongBenchV2Settings


def _build_settings(*, compensate_missing: bool) -> LongBenchV2Settings:
    return LongBenchV2Settings.model_construct(
        truncate_type="last_n_turns",
        rag_topn=1,
        cot=False,
        no_context=False,
        compensate_missing=compensate_missing,
        prompt=None,
        metric_kwargs={"compensate_missing": compensate_missing},
    )


def _build_output(*, output_text: str, answer: str) -> LongBenchV2Output:
    return LongBenchV2Output(
        answer=answer,
        output=output_text,
    )


def _build_config() -> SubtaskConfig:
    return SubtaskConfig(
        name="longbench_v2",
        language="english",
        dataset_filepath=Path("dataset.jsonl"),
        output_filepath=Path("output.jsonl"),
        metric="parsed_answer_match",
        inference_mode="completion",
        settings=None,
    )


def test_parsed_answer_match_compensates_empty_output() -> None:
    metrics = LongBenchV2Metrics(logger=logging.getLogger(__name__))
    settings = _build_settings(compensate_missing=True)
    output = _build_output(output_text="", answer="A")

    score = metrics.eval_parsed_answer_match(
        output=output,
        config=_build_config(),
        settings=settings,
        default_error_message="ERROR",
    )

    assert score == 0.25


@pytest.mark.parametrize(
    ("compensate_missing", "expected"),
    [(True, 0.25), (False, 0.0)],
)
def test_parsed_answer_match_missing_regex(
    compensate_missing: bool, expected: float
) -> None:
    metrics = LongBenchV2Metrics(logger=logging.getLogger(__name__))
    settings = _build_settings(compensate_missing=compensate_missing)
    output = _build_output(output_text="No answer here.", answer="A")

    score = metrics.eval_parsed_answer_match(
        output=output,
        config=_build_config(),
        settings=settings,
        default_error_message="ERROR",
    )

    assert score == expected


def test_parsed_answer_match_invalid_answer_label() -> None:
    metrics = LongBenchV2Metrics(logger=logging.getLogger(__name__))
    settings = _build_settings(compensate_missing=True)
    output = _build_output(output_text="The correct answer is (A)", answer="E")

    score = metrics.eval_parsed_answer_match(
        output=output,
        config=_build_config(),
        settings=settings,
        default_error_message="ERROR",
    )

    assert score == 0.0
