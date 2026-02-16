from dataclasses import dataclass
from logging import getLogger

from benchmarks._core.evaluate.metrics import BaseMetrics
from benchmarks._core.predict.data import Output
from benchmarks.config import SubtaskConfig


@dataclass
class _DummySettings:
    require_reasoning: bool


class _DummyMetrics(BaseMetrics[_DummySettings, Output]):
    _metric_registry = {"dummy": "eval_dummy"}

    def __init__(self) -> None:
        super().__init__(logger=getLogger(__name__))
        self.called = False

    def eval_dummy(
        self,
        output: Output,
        config: SubtaskConfig,
        settings: _DummySettings,
        **kwargs: object,
    ) -> float:
        self.called = True
        return 0.42


def _make_subtask_config() -> SubtaskConfig:
    return SubtaskConfig.model_validate(
        {
            "name": "dummy-subtask",
            "language": "english",
            "dataset_filepath": "dummy.jsonl",
            "output_filepath": "dummy_output.jsonl",
            "metric": "dummy",
            "inference_mode": "chat",
            "settings": {},
        }
    )


def test_eval_returns_zero_when_reasoning_required_and_missing() -> None:
    metrics = _DummyMetrics()
    output = Output.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": "answer",
            "output": "prediction",
            "context_length": 128,
            "output_reasoning": None,
        }
    )

    score = metrics.eval(
        output=output,
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=True),
    )

    assert score == 0.0
    assert metrics.called is False


def test_eval_runs_metric_when_reasoning_required_and_present() -> None:
    metrics = _DummyMetrics()
    output = Output.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": "answer",
            "output": "prediction",
            "context_length": 128,
            "output_reasoning": "<think>...</think>",
        }
    )

    score = metrics.eval(
        output=output,
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=True),
    )

    assert score == 0.42
    assert metrics.called is True


def test_eval_uses_settings_field_even_if_kwargs_disagree() -> None:
    metrics = _DummyMetrics()
    output = Output.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": "answer",
            "output": "prediction",
            "context_length": 128,
            "output_reasoning": None,
        }
    )

    score = metrics.eval(
        output=output,
        config=_make_subtask_config(),
        settings=_DummySettings(require_reasoning=False),
        require_reasoning=True,
    )

    assert score == 0.42
    assert metrics.called is True
