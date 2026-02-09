import logging
from pathlib import Path
import sys
import types

import pytest

benchmarks_src = Path(__file__).resolve().parents[2] / "packages" / "benchmarks" / "src"
sys.path.append(str(benchmarks_src))

benchmarks_module = types.ModuleType("benchmarks")
benchmarks_module.__path__ = [str(benchmarks_src / "benchmarks")]
sys.modules.setdefault("benchmarks", benchmarks_module)

benchmarks_ruler_module = types.ModuleType("benchmarks.ruler")
benchmarks_ruler_module.__path__ = [str(benchmarks_src / "benchmarks" / "ruler")]
sys.modules.setdefault("benchmarks.ruler", benchmarks_ruler_module)

benchmarks_core_module = types.ModuleType("benchmarks._core")
benchmarks_core_module.__path__ = [str(benchmarks_src / "benchmarks" / "_core")]
sys.modules.setdefault("benchmarks._core", benchmarks_core_module)

benchmarks_core_predict_module = types.ModuleType("benchmarks._core.predict")
benchmarks_core_predict_module.__path__ = [
    str(benchmarks_src / "benchmarks" / "_core" / "predict")
]
sys.modules.setdefault("benchmarks._core.predict", benchmarks_core_predict_module)

rapidfuzz_module = types.ModuleType("rapidfuzz")
rapidfuzz_distance_module = types.ModuleType("rapidfuzz.distance")


class _LCSseq:
    @staticmethod
    def similarity(*_args: object, **_kwargs: object) -> int:
        return 0


rapidfuzz_distance_module.LCSseq = _LCSseq
rapidfuzz_module.distance = rapidfuzz_distance_module
sys.modules.setdefault("rapidfuzz", rapidfuzz_module)
sys.modules.setdefault("rapidfuzz.distance", rapidfuzz_distance_module)

from types import SimpleNamespace

from benchmarks.ruler.evaluate import metrics as ruler_metrics
from benchmarks.ruler.evaluate.metrics import RULERMetrics
from benchmarks.ruler.predict.data import RULEROutput
from benchmarks.ruler.settings import RULERSettings


def _make_output(*, output: str, answer: list[str]) -> RULEROutput:
    return RULEROutput(
        id="sample",
        input="prompt",
        answer=answer,
        output=output,
        context_length=128,
        needle_depth=[0.5],
    )


def _make_subtask_config() -> SimpleNamespace:
    return SimpleNamespace(metric="lcs_f1_coverage")


def _make_metrics() -> RULERMetrics:
    return RULERMetrics(logger=logging.getLogger(__name__))


def test_eval_lcs_f1_coverage_returns_zero_for_empty_answer_list() -> None:
    metrics = _make_metrics()
    output = _make_output(output="prediction", answer=[])

    score = metrics.eval_lcs_f1_coverage(
        output=output,
        config=_make_subtask_config(),
        settings=RULERSettings(),
        default_error_message="__DEFAULT_ERROR__",
    )

    assert score == 0.0


def test_eval_lcs_f1_coverage_returns_zero_for_empty_output() -> None:
    metrics = _make_metrics()
    output = _make_output(output="", answer=["reference"])

    score = metrics.eval_lcs_f1_coverage(
        output=output,
        config=_make_subtask_config(),
        settings=RULERSettings(),
        default_error_message="__DEFAULT_ERROR__",
    )

    assert score == 0.0


def test_eval_lcs_f1_coverage_returns_zero_for_empty_ref_string() -> None:
    metrics = _make_metrics()
    output = _make_output(output="prediction", answer=[""])

    score = metrics.eval_lcs_f1_coverage(
        output=output,
        config=_make_subtask_config(),
        settings=RULERSettings(),
        default_error_message="__DEFAULT_ERROR__",
    )

    assert score == 0.0


def test_eval_lcs_f1_coverage_returns_zero_when_lcs_forced_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _make_metrics()
    output = _make_output(output="abc", answer=["ab"])

    monkeypatch.setattr(ruler_metrics.LCSseq, "similarity", lambda *_: 0)

    score = metrics.eval_lcs_f1_coverage(
        output=output,
        config=_make_subtask_config(),
        settings=RULERSettings(),
        default_error_message="__DEFAULT_ERROR__",
    )

    assert score == 0.0
