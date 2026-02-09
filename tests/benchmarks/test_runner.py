from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks._core import runner as runner_module
from benchmarks._core.evaluate.metrics import BaseMetrics
from benchmarks._core.evaluate.table import BaseTable, BaseTableRow, OutputsTable, OutputsTableRow
from benchmarks._core.predict.data import Output
from benchmarks._core.runner import BaseBenchmarkRunner
from benchmarks._core.settings import BaseSettings
from benchmarks.config import BenchmarkConfig, SubtaskConfig, TaskConfig
from llm_inference.base import BaseGenerator


class DummySettings(BaseSettings):
    use_truncate: bool = False


class DummyMetrics(BaseMetrics[DummySettings, Output]):
    pass


@dataclass(frozen=True)
class DummyLeaderboardRow(BaseTableRow):
    score: float


class DummyGenerator(BaseGenerator):
    def _count_tokens(self, input: Any, **kwargs: Any) -> int:
        return 1

    def _chat(self, *, conversations: list[Any], **kwargs: Any) -> list[Any]:
        return []

    def _completion(self, *, prompts: list[Any], **kwargs: Any) -> list[Any]:
        return []


class DummyRunner(
    BaseBenchmarkRunner[DummySettings, Output, OutputsTableRow, DummyLeaderboardRow]
):
    def _build_metrics(self) -> DummyMetrics:
        return DummyMetrics(self.logger)

    @property
    def settings_model(self) -> type[DummySettings]:
        return DummySettings

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: DummySettings,
        batchsize: int,
    ) -> list[Output]:
        return [
            Output(
                id="sample",
                input="input",
                answer="answer",
                output="output",
                context_length=123,
            )
        ]

    def _evaluate_subtask(
        self,
        *,
        model_name: str,
        task: str,
        config: SubtaskConfig,
        settings: DummySettings,
        output: Output,
        **kwargs: Any,
    ) -> OutputsTableRow:
        return OutputsTableRow(
            model_name=model_name,
            id=output.id,
            task=task,
            subtask=config.name,
            language=config.language,
            context_length=output.context_length,
            input=output.input,
            answer=str(output.answer),
            score=1.0,
            output=output.output,
            output_reasoning=output.output_reasoning,
        )

    def _to_outputs_table(
        self, name: str, rows: list[OutputsTableRow]
    ) -> OutputsTable[OutputsTableRow]:
        return OutputsTable(name=name, rows=rows)

    def _make_leaderboard_table(
        self, outputs: list[OutputsTableRow]
    ) -> BaseTable[DummyLeaderboardRow]:
        return BaseTable(
            name="leaderboard",
            rows=[DummyLeaderboardRow(model_name=outputs[0].model_name, score=1.0)],
        )


def _make_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="demo",
        output_root=Path("/tmp/output"),
        tasks=[
            TaskConfig(
                name="task",
                subtasks=[
                    SubtaskConfig(
                        name="subtask",
                        language="english",
                        dataset_filepath=Path("/tmp/data.json"),
                        output_filepath=Path("/tmp/output.json"),
                        metric="dummy",
                        inference_mode="chat",
                        settings={},
                    )
                ],
            )
        ],
    )


def _make_runner() -> DummyRunner:
    return DummyRunner(logger=__import__("logging").getLogger(__name__))


def _make_generator() -> DummyGenerator:
    return DummyGenerator(
        model_name="demo-model",
        max_context_length=4096,
        max_output_tokens=10,
        logger=__import__("logging").getLogger(__name__),
    )


def test_runner_saves_and_pushes_tables(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_save_to_local(*, tables: Any, output_root: Path, format: str) -> None:
        calls["save"] = {
            "tables": tables,
            "output_root": output_root,
            "format": format,
        }

    def fake_push_to_wandb(*, tables: Any) -> None:
        calls.setdefault("push", []).append(tables)

    monkeypatch.setattr(runner_module, "save_to_local", fake_save_to_local)
    monkeypatch.setattr(runner_module, "push_to_wandb", fake_push_to_wandb)

    runner = _make_runner()
    runner.run(
        generator=_make_generator(),
        generation_kwargs={},
        config=_make_config(),
        batchsize=1,
        log_wandb=True,
    )

    assert calls["save"]["output_root"] == Path("/tmp/output")
    assert calls["save"]["format"] == "jsonl"
    assert calls["save"]["tables"] is calls["push"][0]
    assert len(calls["push"]) == 1


def test_runner_skips_wandb_when_disabled(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_save_to_local(*, tables: Any, output_root: Path, format: str) -> None:
        calls["save"] = {
            "tables": tables,
            "output_root": output_root,
            "format": format,
        }

    def fake_push_to_wandb(*, tables: Any) -> None:
        calls.setdefault("push", []).append(tables)

    monkeypatch.setattr(runner_module, "save_to_local", fake_save_to_local)
    monkeypatch.setattr(runner_module, "push_to_wandb", fake_push_to_wandb)

    runner = _make_runner()
    runner.run(
        generator=_make_generator(),
        generation_kwargs={},
        config=_make_config(),
        batchsize=1,
        log_wandb=False,
    )

    assert calls["save"]["output_root"] == Path("/tmp/output")
    assert calls["save"]["format"] == "jsonl"
    assert "push" not in calls
