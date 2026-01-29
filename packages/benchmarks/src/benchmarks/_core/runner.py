from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generic

from llm_inference.base import BaseGenerator
from pydantic import ValidationError

from ..config import BenchmarkConfig, SubtaskConfig
from .evaluate import (
    Bin,
    mean_score_by_group_and_context_bin,
)
from .evaluate.table import (
    BenchmarkResults,
    LeaderboardTableType,
    MeanScoreTable,
    MeanScoreTableRow,
    OutputsTable,
    OutputsTableRowType,
)
from .predict.data import OutputType
from .save_table import push_to_wandb, save_to_local
from .settings import SettingsType


class BaseBenchmarkRunner(
    ABC, Generic[SettingsType, OutputType, OutputsTableRowType, LeaderboardTableType]
):
    def __init__(self, logger: Logger, bins: list[Bin] | None = None):
        self.logger = logger
        if bins:
            self.bins = sorted(bins, key=lambda b: b.upper)
        else:
            self.bins = [
                Bin(upper=length, label=f"~{length}")
                for length in [2**i * 1024 for i in range(2, 8)]
            ]

    @property
    @abstractmethod
    def settings_model(self) -> type[SettingsType]:
        raise NotImplementedError

    def _validate_settings(self, subtask_config: SubtaskConfig) -> SettingsType:
        try:
            # Pydantic v2
            return self.settings_model.model_validate(subtask_config.settings)
        except ValidationError as e:
            # どの subtask の設定が壊れてるか分かるようにする
            raise ValueError(
                f"Invalid settings for subtask='{subtask_config.name}': {e}"
            ) from e

    @abstractmethod
    def _make_leaderboard_table(
        self, outputs: list[OutputsTableRowType]
    ) -> LeaderboardTableType: ...

    @abstractmethod
    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: SettingsType,
        batchsize: int,
    ) -> list[OutputType]: ...

    @abstractmethod
    def _to_output_row(
        self, *, task: str, subtask: str, output: OutputType
    ) -> OutputsTableRowType: ...

    @abstractmethod
    def _to_outputs_table(
        self, name: str, rows: list[OutputsTableRowType]
    ) -> OutputsTable[OutputsTableRowType]: ...

    def run(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: BenchmarkConfig,
        batchsize: int,
        log_wandb: bool,
    ) -> None:
        mean_score_by_subtask: list[MeanScoreTableRow] = []
        mean_score_by_task: list[MeanScoreTableRow] = []
        mean_score_by_language: list[MeanScoreTableRow] = []

        all_outputs: list[OutputsTableRowType] = []
        for task_config in config.tasks:
            outputs_all_subtask: list[OutputsTableRowType] = []
            for subtask_config in task_config.subtasks:
                settings = self._validate_settings(subtask_config)
                outputs: list[OutputType] = self._run_subtask(
                    generator=generator,
                    generation_kwargs=generation_kwargs,
                    config=subtask_config,
                    settings=settings,
                    batchsize=batchsize,
                )

                output_rows = [
                    self._to_output_row(
                        output=out,
                        task=task_config.name,
                        subtask=subtask_config.name,
                    )
                    for out in outputs
                ]

                # subtask ごとに context length の bin ごとの平均スコアを算出
                mean_score_by_subtask += mean_score_by_group_and_context_bin(
                    rows=output_rows,
                    group_by="subtask",
                    context_bins=self.bins,
                )
                outputs_all_subtask += output_rows

            # task ごとに context length の bin ごとの平均スコアを算出
            mean_score_by_task += mean_score_by_group_and_context_bin(
                rows=outputs_all_subtask,
                group_by="task",
                context_bins=self.bins,
            )
            all_outputs += outputs_all_subtask

        # language ごとに context length の bin ごとの平均スコアを算出
        mean_score_by_language += mean_score_by_group_and_context_bin(
            rows=all_outputs,
            group_by="language",
            context_bins=self.bins,
        )

        # Table を作成
        results = BenchmarkResults(
            outputs_table=self._to_outputs_table(
                name=f"{config.name}_outputs_table",
                rows=all_outputs,
            ),
            mean_score_by_subtask=MeanScoreTable(
                name=f"{config.name}_mean_score_by_subtask", rows=mean_score_by_subtask
            ),
            mean_score_by_task=MeanScoreTable(
                name=f"{config.name}_mean_score_by_task", rows=mean_score_by_task
            ),
            mean_score_by_language=MeanScoreTable(
                name=f"{config.name}_mean_score_by_language",
                rows=mean_score_by_language,
            ),
            leaderboard_table=self._make_leaderboard_table(outputs=all_outputs),
        )

        # Save
        save_to_local(
            tables=results.tables, output_root=config.output_root, format="jsonl"
        )

        if log_wandb:
            push_to_wandb(tables=results.tables)
