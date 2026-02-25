from dataclasses import dataclass
from logging import getLogger

from benchmarks._core.evaluate.metrics import BaseMetrics
from benchmarks._core.predict.data import Output
from benchmarks.config import SubtaskConfig


@dataclass
class _DummySettings:
    require_reasoning: bool


class _DummyMetrics(BaseMetrics[_DummySettings, Output]):
    _metric_registry = {"dummy_metric": "eval_dummy_metric"}

    def __init__(self) -> None:
        super().__init__(logger=getLogger(__name__))
        self.eval_dummy_metric_called = False

    def eval_dummy_metric(
        self,
        output: Output,
        config: SubtaskConfig,
        settings: _DummySettings,
        **kwargs: object,
    ) -> float:
        self.eval_dummy_metric_called = True
        return 0.7


def _make_subtask_config(metric: str = "dummy_metric") -> SubtaskConfig:
    return SubtaskConfig.model_validate(
        {
            "name": "dummy-subtask",
            "language": "english",
            "dataset_filepath": "dummy.jsonl",
            "output_filepath": "dummy_output.jsonl",
            "metric": metric,
            "inference_mode": "chat",
            "settings": {},
        }
    )


def _make_output(output_reasoning: str | None) -> Output:
    return Output.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": "answer",
            "output": "prediction",
            "context_length": 128,
            "output_reasoning": output_reasoning,
        }
    )


def test_eval_returns_zero_when_reasoning_required_and_missing() -> None:
    metrics = _DummyMetrics()

    score = metrics.eval(
        output=_make_output(output_reasoning=None),
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=True),
    )

    assert score == 0.0
    assert metrics.eval_dummy_metric_called is False


def test_eval_dispatches_to_registry_method_when_reasoning_present() -> None:
    metrics = _DummyMetrics()

    score = metrics.eval(
        output=_make_output(output_reasoning="<think>...</think>"),
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=True),
    )

    assert score == 0.7
    assert metrics.eval_dummy_metric_called is True


def test_settings_require_reasoning_takes_precedence_over_kwargs() -> None:
    metrics = _DummyMetrics()

    score = metrics.eval(
        output=_make_output(output_reasoning="<think>...</think>"),
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=False),
        require_reasoning=True,
    )

    assert score == 0.7
    assert metrics.eval_dummy_metric_called is True


def test_metric_key_resolves_to_expected_method_name() -> None:
    metrics = _DummyMetrics()
    config = _make_subtask_config(metric="dummy_metric")

    method_name = metrics._metric_registry[config.metric]

    assert method_name == "eval_dummy_metric"
    assert getattr(metrics, method_name).__name__ == "eval_dummy_metric"
