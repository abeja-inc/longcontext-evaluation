from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

from benchmarks._core.evaluate.metrics import BaseMetrics
from benchmarks._core.evaluate.table import (
    BaseTable,
    OutputsTable,
    OutputsTableRow,
)
from benchmarks._core.predict.data import Output
from benchmarks._core.runner import BaseBenchmarkRunner
from benchmarks._core.settings import BaseSettings
from benchmarks.config import BenchmarkConfig
from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Prompt, Response


class _DummyGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__(
            model_name="dummy-model",
            max_context_length=4096,
            max_output_tokens=128,
            logger=getLogger(__name__),
        )
        self.reasoning_parser = "dummy-parser"

    def _count_tokens(self, input: Prompt | Conversation, **kwargs: Any) -> int:
        return 1

    def _chat(
        self, *, conversations: list[Conversation], **kwargs: Any
    ) -> list[Response]:
        return [
            Response.model_validate(
                {"outputs": [{"content": "ok", "reasoning_content": None}]}
            )
            for _ in conversations
        ]

    def _completion(self, *, prompts: list[Prompt], **kwargs: Any) -> list[Response]:
        return [
            Response.model_validate(
                {"outputs": [{"content": "ok", "reasoning_content": None}]}
            )
            for _ in prompts
        ]


@dataclass(frozen=True)
class _DummyOutputsRow(OutputsTableRow):
    pass


@dataclass(frozen=True)
class _DummyLeaderboardRow:
    model_name: str


class _DummyMetrics(BaseMetrics[BaseSettings, Output]):
    _metric_registry = {"dummy": "eval_dummy"}

    def __init__(self) -> None:
        super().__init__(logger=getLogger(__name__))

    def eval_dummy(
        self,
        output: Output,
        config: Any,
        settings: BaseSettings,
        **kwargs: Any,
    ) -> float:
        return 1.0


class _DummyRunner(
    BaseBenchmarkRunner[BaseSettings, Output, _DummyOutputsRow, _DummyLeaderboardRow]
):
    def __init__(self) -> None:
        super().__init__(logger=getLogger(__name__))
        self.received_require_reasoning: list[bool] = []
        self.received_scores: list[float] = []

    @property
    def settings_model(self) -> type[BaseSettings]:
        return BaseSettings

    def _build_metrics(self) -> BaseMetrics[BaseSettings, Output]:
        return _DummyMetrics()

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: Any,
        settings: BaseSettings,
        batchsize: int,
    ) -> list[Output]:
        return [
            Output.model_validate(
                {
                    "id": "1",
                    "input": "q",
                    "answer": "a",
                    "output": "pred",
                    "context_length": 10,
                    "output_reasoning": None,
                }
            )
        ]

    def _evaluate_subtask(
        self,
        *,
        model_name: str,
        task: str,
        config: Any,
        settings: BaseSettings,
        output: Output,
        **kwargs: Any,
    ) -> _DummyOutputsRow:
        self.received_require_reasoning.append("require_reasoning" in kwargs)
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        self.received_scores.append(score)
        return _DummyOutputsRow(
            model_name=model_name,
            id=output.id,
            task=task,
            subtask=config.name,
            language=config.language,
            context_length=output.context_length,
            input=output.input,
            answer=str(output.answer),
            score=score,
            output=output.output,
            output_reasoning=output.output_reasoning,
        )

    def _to_outputs_table(
        self, name: str, rows: list[_DummyOutputsRow]
    ) -> OutputsTable[_DummyOutputsRow]:
        return OutputsTable(name=name, rows=rows)

    def _make_leaderboard_table(
        self, outputs: list[_DummyOutputsRow]
    ) -> BaseTable[_DummyLeaderboardRow]:
        return BaseTable(name="dummy", rows=[_DummyLeaderboardRow(model_name="dummy")])


def _make_benchmark_config(
    tmp_path: Path, *, require_reasoning: bool
) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "name": "dummy-benchmark",
            "output_root": str(tmp_path / "out"),
            "tasks": [
                {
                    "name": "dummy-task",
                    "subtasks": [
                        {
                            "name": "dummy-subtask",
                            "language": "english",
                            "dataset_filepath": str(tmp_path / "dataset.jsonl"),
                            "output_filepath": str(tmp_path / "output.jsonl"),
                            "metric": "dummy",
                            "inference_mode": "chat",
                            "settings": {
                                "use_truncate": False,
                                "require_reasoning": require_reasoning,
                            },
                        }
                    ],
                }
            ],
        }
    )


def test_run_does_not_forward_require_reasoning_kwarg(tmp_path: Path) -> None:
    runner = _DummyRunner()
    runner.run(
        generator=_DummyGenerator(),
        generation_kwargs={},
        config=_make_benchmark_config(tmp_path, require_reasoning=True),
        batchsize=1,
        log_wandb=False,
    )

    assert runner.received_require_reasoning == [False]


def test_run_scoring_still_uses_settings_require_reasoning_false(
    tmp_path: Path,
) -> None:
    runner = _DummyRunner()
    runner.run(
        generator=_DummyGenerator(),
        generation_kwargs={},
        config=_make_benchmark_config(tmp_path, require_reasoning=False),
        batchsize=1,
        log_wandb=False,
    )

    assert runner.received_require_reasoning == [False]
    assert runner.received_scores == [1.0]


def test_run_scoring_still_uses_settings_require_reasoning_true(tmp_path: Path) -> None:
    runner = _DummyRunner()
    runner.run(
        generator=_DummyGenerator(),
        generation_kwargs={},
        config=_make_benchmark_config(tmp_path, require_reasoning=True),
        batchsize=1,
        log_wandb=False,
    )

    assert runner.received_require_reasoning == [False]
    assert runner.received_scores == [0.0]
