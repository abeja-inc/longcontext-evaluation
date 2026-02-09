import logging
import sys
import types
import logging
from pathlib import Path

from benchmarks.longbench_v2.evaluate.metrics import LongBenchV2Metrics
from benchmarks.longbench_v2.predict.data import Domain, Difficulty, Length, LongBenchV2Output, SubDomain
from benchmarks.mrcr.evaluate.metrics import OpenAIMRCRMetrics
from benchmarks.mrcr.predict.data import OpenAIMRCROutput
from benchmarks.mrcr.settings import OpenAIMRCRSettings
from benchmarks.ruler.evaluate.metrics import RULERMetrics
from benchmarks.ruler.predict.data import RULEROutput
from benchmarks.ruler.settings import RULERSettings

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

def test_longbench_parsed_answer_match() -> None:
    metrics = LongBenchV2Metrics(logger=logging.getLogger("test"))
    class DummySettings:
        metric_kwargs: dict[str, object] = {}

    settings = DummySettings()
    output = LongBenchV2Output(
        id="1",
        input="input",
        answer="A",
        output="The correct answer is (A)",
        context_length=10,
        output_reasoning=None,
        difficulty=Difficulty.EASY,
        length=Length.SHORT,
        domain=Domain.SINGLE_DOCUMENT_QA,
        sub_domain=SubDomain.ACADEMIC,
    )
    score = metrics.eval_parsed_answer_match(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score == 1.0

    output.output = ""
    score = metrics.eval_parsed_answer_match(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score == 0.0


def test_mrcr_prefix_match_similarity() -> None:
    metrics = OpenAIMRCRMetrics(logger=logging.getLogger("test"))
    settings = OpenAIMRCRSettings(metric_kwargs={})
    output = OpenAIMRCROutput(
        id="1",
        input="input",
        answer="PREFIXanswer",
        output="PREFIXanswer",
        context_length=10,
        output_reasoning=None,
        random_string_to_prepend="PREFIX",
        n_needles=1,
        desired_msg_index=0,
        total_messages=1,
    )
    score = metrics.eval_prefix_match_similarity(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score == 1.0

    output.output = "[ERROR]"
    score = metrics.eval_prefix_match_similarity(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score == 0.0


def test_ruler_metrics_boundaries() -> None:
    metrics = RULERMetrics(logger=logging.getLogger("test"))
    settings = RULERSettings(metric_kwargs={})
    output = RULEROutput(
        id="1",
        input="input",
        answer=["ABC", "DEF"],
        output="ABC",
        context_length=10,
        output_reasoning=None,
        needle_depth=[10.0],
    )
    score_any = metrics.eval_substr_any(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score_any == 1.0

    score_cov = metrics.eval_substr_coverage(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score_cov == 0.5

    score_lcs = metrics.eval_lcs_f1_max(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score_lcs == 1.0

    output.output = "[ERROR]"
    score_error = metrics.eval_lcs_f1_coverage(
        output=output,
        config=None,  # type: ignore[arg-type]
        settings=settings,
        default_error_message="[ERROR]",
    )
    assert score_error == 0.0
