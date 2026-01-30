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
from .evaluate.metrics import BaseMetrics
from .evaluate.table import (
    BaseTable,
    BenchmarkResults,
    LeaderboardTableRowType,
    MeanScoreByLengthTable,
    MeanScoreByLengthTableRow,
    OutputsTable,
    OutputsTableRowType,
)
from .predict.data import OutputType
from .save_table import push_to_wandb, save_to_local
from .settings import SettingsType


class BaseBenchmarkRunner(
    ABC,
    Generic[SettingsType, OutputType, OutputsTableRowType, LeaderboardTableRowType],
):
    def __init__(
        self,
        logger: Logger,
        bins: list[Bin] | None = None,
    ):
        self.logger = logger
        if bins:
            self.bins = sorted(bins, key=lambda b: b.upper)
        else:
            self.bins = [
                Bin(upper=length, label=f"~{length}")
                for length in [2**i * 1024 for i in range(2, 8)]
            ]
        self._metrics: BaseMetrics[SettingsType, OutputType] | None = None

    @abstractmethod
    def _build_metrics(self) -> BaseMetrics[SettingsType, OutputType]: ...

    @property
    def metrics(self) -> BaseMetrics[SettingsType, OutputType]:
        if self._metrics is None:
            self._metrics = self._build_metrics()
        return self._metrics

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
    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: SettingsType,
        batchsize: int,
    ) -> list[OutputType]: ...

    @abstractmethod
    def _evaluate_subtask(
        self,
        *,
        model_name: str,
        task: str,
        config: SubtaskConfig,
        settings: SettingsType,
        output: OutputType,
        **kwargs: Any,
    ) -> OutputsTableRowType: ...

    @abstractmethod
    def _to_outputs_table(
        self, name: str, rows: list[OutputsTableRowType]
    ) -> OutputsTable[OutputsTableRowType]: ...

    @abstractmethod
    def _make_leaderboard_table(
        self, outputs: list[OutputsTableRowType]
    ) -> BaseTable[LeaderboardTableRowType]: ...

    def _make_additional_tables(
        self, outputs: list[OutputsTableRowType]
    ) -> list[BaseTable[Any]]:
        return []

    def run(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: BenchmarkConfig,
        batchsize: int,
        log_wandb: bool,
    ) -> None:
        mean_score_by_subtask: list[MeanScoreByLengthTableRow] = []
        mean_score_by_task: list[MeanScoreByLengthTableRow] = []
        mean_score_by_language: list[MeanScoreByLengthTableRow] = []

        all_outputs: list[OutputsTableRowType] = []
        for task_config in config.tasks:
            outputs_all_subtask: list[OutputsTableRowType] = []
            for subtask_config in task_config.subtasks:
                settings = self._validate_settings(subtask_config)
                # Predict
                outputs: list[OutputType] = self._run_subtask(
                    generator=generator,
                    generation_kwargs=generation_kwargs,
                    config=subtask_config,
                    settings=settings,
                    batchsize=batchsize,
                )

                # Evaluate
                output_rows = [
                    self._evaluate_subtask(
                        model_name=generator.model_name,
                        output=out,
                        task=task_config.name,
                        subtask=subtask_config.name,
                        config=subtask_config,
                        settings=settings,
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
        results = BenchmarkResults[OutputsTableRowType, LeaderboardTableRowType](
            outputs_table=self._to_outputs_table(
                name=f"{config.name}_outputs_table",
                rows=all_outputs,
            ),
            mean_score_by_subtask=MeanScoreByLengthTable(
                name=f"{config.name}_mean_score_by_subtask", rows=mean_score_by_subtask
            ),
            mean_score_by_task=MeanScoreByLengthTable(
                name=f"{config.name}_mean_score_by_task", rows=mean_score_by_task
            ),
            mean_score_by_language=MeanScoreByLengthTable(
                name=f"{config.name}_mean_score_by_language",
                rows=mean_score_by_language,
            ),
            leaderboard_table=self._make_leaderboard_table(outputs=all_outputs),
            additional_tables=self._make_additional_tables(outputs=all_outputs),
        )

        # Save
        save_to_local(
            tables=results.tables, output_root=config.output_root, format="jsonl"
        )

        if log_wandb:
            push_to_wandb(tables=results.tables)
