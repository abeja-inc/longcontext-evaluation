from __future__ import annotations

import logging

from benchmarks.longbench_v2.evaluate.metrics import LongBenchV2Metrics
from benchmarks.longbench_v2.predict.data import Domain, Difficulty, Length, LongBenchV2Output, SubDomain
from benchmarks.mrcr.evaluate.metrics import OpenAIMRCRMetrics
from benchmarks.mrcr.predict.data import OpenAIMRCROutput
from benchmarks.mrcr.settings import OpenAIMRCRSettings
from benchmarks.ruler.evaluate.metrics import RULERMetrics
from benchmarks.ruler.predict.data import RULEROutput
from benchmarks.ruler.settings import RULERSettings


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
