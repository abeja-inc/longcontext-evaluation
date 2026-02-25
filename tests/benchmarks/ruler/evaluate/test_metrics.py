import logging
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARKS_SRC = REPO_ROOT / "packages" / "benchmarks" / "src" / "benchmarks"
LLM_INFERENCE_SRC = REPO_ROOT / "packages" / "llm_inference" / "src"

if str(LLM_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_INFERENCE_SRC))

if "benchmarks" not in sys.modules:
    benchmarks_stub = types.ModuleType("benchmarks")
    benchmarks_stub.__path__ = [str(BENCHMARKS_SRC)]
    sys.modules["benchmarks"] = benchmarks_stub

if "benchmarks.ruler" not in sys.modules:
    ruler_stub = types.ModuleType("benchmarks.ruler")
    ruler_stub.__path__ = [str(BENCHMARKS_SRC / "ruler")]
    sys.modules["benchmarks.ruler"] = ruler_stub

if "rapidfuzz.distance" not in sys.modules:
    rapidfuzz_stub = types.ModuleType("rapidfuzz")
    rapidfuzz_distance_stub = types.ModuleType("rapidfuzz.distance")

    class _LCSseq:
        @staticmethod
        def similarity(a: str, b: str) -> int:
            dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i, ch_a in enumerate(a, start=1):
                for j, ch_b in enumerate(b, start=1):
                    if ch_a == ch_b:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[-1][-1]

    rapidfuzz_distance_stub.LCSseq = _LCSseq
    rapidfuzz_stub.distance = rapidfuzz_distance_stub
    sys.modules["rapidfuzz"] = rapidfuzz_stub
    sys.modules["rapidfuzz.distance"] = rapidfuzz_distance_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *args, **kwargs):
        return iterable

    tqdm_stub.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_stub


from benchmarks.config import SubtaskConfig
from benchmarks.ruler.evaluate.metrics import RULERMetrics
from benchmarks.ruler.predict.data import RULEROutput
from benchmarks.ruler.settings import RULERSettings


DEFAULT_ERROR_MESSAGE = "<default-error>"


def _make_output(*, output: str, answer: list[str]) -> RULEROutput:
    return RULEROutput(
        id="sample-1",
        input="prompt",
        answer=answer,
        output=output,
        context_length=8,
        needle_depth=[10.0],
    )


def _make_metrics() -> RULERMetrics:
    return RULERMetrics(logger=logging.getLogger(__name__))


def _make_config() -> SubtaskConfig:
    return SubtaskConfig(
        name="subtask",
        language="english",
        dataset_filepath=Path("dataset.jsonl"),
        output_filepath=Path("output.jsonl"),
        metric="substr_any",
        inference_mode="completion",
    )


class TestEvalSubstrAny:
    def test_case_insensitive_match_returns_one(self):
        metrics = _make_metrics()
        output = _make_output(output="Answer: Hello WORLD", answer=["world"])

        score = metrics._eval_substr_any(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 1.0

    def test_no_match_returns_zero(self):
        metrics = _make_metrics()
        output = _make_output(output="alpha beta", answer=["gamma", "delta"])

        score = metrics._eval_substr_any(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0

    @pytest.mark.parametrize("pred", ["", DEFAULT_ERROR_MESSAGE])
    def test_empty_or_default_error_output_returns_zero(self, pred):
        metrics = _make_metrics()
        output = _make_output(output=pred, answer=["alpha"])

        score = metrics._eval_substr_any(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0


class TestEvalSubstrCoverage:
    @pytest.mark.parametrize(
        ("pred", "refs", "expected"),
        [
            ("alpha", ["alpha", "beta"], 0.5),
            ("a b", ["a", "b", "z"], 2 / 3),
        ],
    )
    def test_mixed_hits_compute_ratio(self, pred, refs, expected):
        metrics = _make_metrics()
        output = _make_output(output=pred, answer=refs)

        score = metrics._eval_substr_coverage(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == pytest.approx(expected)

    @pytest.mark.parametrize("pred", ["", DEFAULT_ERROR_MESSAGE])
    def test_empty_or_default_error_output_returns_zero(self, pred):
        metrics = _make_metrics()
        output = _make_output(output=pred, answer=["alpha", "beta"])

        score = metrics._eval_substr_coverage(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0


class TestEvalLcsF1Max:
    def test_exact_match_returns_one(self):
        metrics = _make_metrics()
        output = _make_output(output="abcd", answer=["abcd", "zzzz"])

        score = metrics._eval_lcs_f1_max(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 1.0

    def test_partial_overlap_returns_score_between_zero_and_one(self):
        metrics = _make_metrics()
        output = _make_output(output="abcd", answer=["abxy"])

        score = metrics._eval_lcs_f1_max(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == pytest.approx(0.5)
        assert 0.0 < score < 1.0

    @pytest.mark.parametrize("pred", ["", DEFAULT_ERROR_MESSAGE])
    def test_empty_or_default_error_output_returns_zero(self, pred):
        metrics = _make_metrics()
        output = _make_output(output=pred, answer=["abcd"])

        score = metrics._eval_lcs_f1_max(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0


class TestEvalLcsF1Coverage:
    def test_returns_average_f1_across_refs(self):
        metrics = _make_metrics()
        output = _make_output(output="abcd", answer=["abcd", "abxy"])

        score = metrics._eval_lcs_f1_coverage(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == pytest.approx(0.75)

    def test_empty_ref_list_returns_zero(self):
        metrics = _make_metrics()
        output = _make_output(output="abcd", answer=[])

        score = metrics._eval_lcs_f1_coverage(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0

    @pytest.mark.parametrize("pred", ["", DEFAULT_ERROR_MESSAGE])
    def test_empty_or_default_error_output_returns_zero(self, pred):
        metrics = _make_metrics()
        output = _make_output(output=pred, answer=["abcd", "abxy"])

        score = metrics._eval_lcs_f1_coverage(
            default_error_message=DEFAULT_ERROR_MESSAGE,
            output=output,
        )

        assert score == 0.0


def test_eval_substr_any_forwards_settings_metric_kwargs(monkeypatch):
    metrics = _make_metrics()
    output = _make_output(output="alpha", answer=["alpha"])
    settings = RULERSettings(metric_kwargs={"foo": "bar"})
    config = _make_config()
    captured_kwargs = {}

    def _fake_eval_substr_any(**kwargs):
        captured_kwargs.update(kwargs)
        return 1.0

    monkeypatch.setattr(metrics, "_eval_substr_any", _fake_eval_substr_any)

    score = metrics.eval_substr_any(
        output=output,
        config=config,
        settings=settings,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        passed_from_eval=True,
    )

    assert score == 1.0
    assert captured_kwargs["foo"] == "bar"
    assert captured_kwargs["passed_from_eval"] is True
    assert captured_kwargs["default_error_message"] == DEFAULT_ERROR_MESSAGE
    assert captured_kwargs["output"] == output
