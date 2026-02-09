from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from benchmarks._core.evaluate.mean_score import Bin, mean_score_by_group_and_context_bin
from benchmarks._core.evaluate.table import BaseTable, OutputsTable, OutputsTableRow
from benchmarks._core.runner import BaseBenchmarkRunner
from benchmarks.config import BenchmarkConfig, SubtaskConfig, TaskConfig


class DummySettings(BaseModel):
    required: int


class DummyRunner(BaseBenchmarkRunner[DummySettings, int, OutputsTableRow, OutputsTableRow]):
    @property
    def settings_model(self) -> type[DummySettings]:
        return DummySettings

    def _build_metrics(self):
        raise NotImplementedError

    def _run_subtask(
        self,
        generator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: DummySettings,
        batchsize: int,
    ) -> list[int]:
        return [1]

    def _evaluate_subtask(
        self,
        *,
        model_name: str,
        task: str,
        config: SubtaskConfig,
        settings: DummySettings,
        output: int,
        **kwargs: Any,
    ) -> OutputsTableRow:
        return OutputsTableRow(
            id="1",
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            context_length=10,
            input="input",
            answer="answer",
            score=1.0,
            output="output",
            output_reasoning=None,
        )

    def _to_outputs_table(self, name: str, rows: list[OutputsTableRow]) -> OutputsTable:
        return OutputsTable(name=name, rows=rows)

    def _make_leaderboard_table(self, outputs: list[OutputsTableRow]) -> BaseTable[OutputsTableRow]:
        return BaseTable(name="leaderboard", rows=outputs)


def _dummy_config(tmp_path: Path) -> BenchmarkConfig:
    subtask = SubtaskConfig(
        name="subtask",
        language="english",
        dataset_filepath=tmp_path / "data.json",
        output_filepath=tmp_path / "out.json",
        metric="dummy",
        inference_mode="chat",
        settings={"required": 1},
    )
    task = TaskConfig(name="task", subtasks=[subtask])
    return BenchmarkConfig(name="bench", output_root=tmp_path, tasks=[task])


def test_validate_settings_error_message() -> None:
    runner = DummyRunner(logger=logging.getLogger("test"), bins=[Bin(upper=10, label="~10")])
    subtask = SubtaskConfig(
        name="bad",
        language="english",
        dataset_filepath=Path("data.json"),
        output_filepath=Path("out.json"),
        metric="dummy",
        inference_mode="chat",
        settings={},
    )
    with pytest.raises(ValueError, match="Invalid settings for subtask='bad'"):
        runner._validate_settings(subtask)


def test_runner_calls_mean_score_bins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_mean_score(*, rows, group_by, context_bins, **kwargs):
        calls.append(group_by)
        return mean_score_by_group_and_context_bin(
            rows=rows, group_by=group_by, context_bins=context_bins
        )

    monkeypatch.setattr(
        "benchmarks._core.runner.mean_score_by_group_and_context_bin",
        fake_mean_score,
    )
    monkeypatch.setattr("benchmarks._core.runner.save_to_local", lambda **kwargs: None)
    monkeypatch.setattr("benchmarks._core.runner.push_to_wandb", lambda **kwargs: None)

    runner = DummyRunner(logger=logging.getLogger("test"), bins=[Bin(upper=10, label="~10")])
    config = _dummy_config(tmp_path)

    @dataclass
    class DummyGenerator:
        model_name: str = "dummy"
        default_too_long_input_error_message: str = "[ERROR]"

    runner.run(
        generator=DummyGenerator(),
        generation_kwargs={},
        config=config,
        batchsize=1,
        log_wandb=False,
    )

    assert calls == ["subtask", "task", "language"]
