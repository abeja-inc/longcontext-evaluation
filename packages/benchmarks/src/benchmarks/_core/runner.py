from abc import ABC, abstractmethod
from logging import Logger

from llm_inference.base import BaseGenerator

from ..config import BenchmarkConfig, SubtaskConfig
from .evaluate import (
    BaseTable,
    MeanScoreTable,
    MeanScoreTableRow,
    OutputsTable,
    OutputsTableRow,
)
from .evaluate.table import Bin, mean_score_by_group_and_context_bin
from .logging import log_results_local, log_results_wandb
from .predict import Output


class BaseBenchmarkRunner(ABC):
    def __init__(self, logger: Logger, bins: list[Bin] | None = None):
        self.logger = logger
        if bins:
            self.bins = sorted(bins, key=lambda b: b.upper)
        else:
            self.bins = [
                Bin(upper=length, label=f"~{length}")
                for length in [2**i * 1024 for i in range(2, 8)]
            ]

    @abstractmethod
    def make_leaderboard_table(self, outputs: list[OutputsTableRow]) -> BaseTable: ...

    @abstractmethod
    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        batchsize: int,
    ) -> list[Output]: ...

    def run(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: BenchmarkConfig,
        batchsize: int,
        save_local: bool,
        log_wandb: bool,
    ) -> None:
        mean_score_by_subtask: list[MeanScoreTableRow] = []
        mean_score_by_task: list[MeanScoreTableRow] = []
        mean_score_by_language: list[MeanScoreTableRow] = []

        all_outputs: list[OutputsTableRow] = []
        for task_config in config.tasks:
            outputs_all_subtask: list[OutputsTableRow] = []
            for subtask_config in task_config.subtasks:
                outputs: list[Output] = self._run_subtask(
                    generator=generator,
                    generation_kwargs=generation_kwargs,
                    config=subtask_config,
                    batchsize=batchsize,
                )

                output_rows = [
                    OutputsTableRow(
                        task=task_config.name,
                        subtask=subtask_config.name,
                        **out.asdict(),
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
        tables: list[BaseTable] = []
        tables.append(
            OutputsTable(
                name=f"{config.name}_outputs_table",
                rows=all_outputs,
            )
        )
        tables.append(
            MeanScoreTable(
                name=f"{config.name}_mean_score_by_subtask", rows=mean_score_by_subtask
            )
        )
        tables.append(
            MeanScoreTable(
                name=f"{config.name}_mean_score_by_task", rows=mean_score_by_task
            )
        )
        tables.append(
            MeanScoreTable(
                name=f"{config.name}_mean_score_by_language",
                rows=mean_score_by_language,
            )
        )

        # leaderboard_table を追加
        tables.append(self.make_leaderboard_table(outputs=all_outputs))

        if save_local:
            log_results_local(
                tables=tables,
                output_root=output_root,
                format="jsonl",
            )

        if log_wandb:
            log_results_wandb(tables=tables)
