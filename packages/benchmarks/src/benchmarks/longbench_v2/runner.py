from llm_inference.base import BaseGenerator

from .._core.evaluate import OutputsTable
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate import (
    LongBenchV2LeaderBoardTable,
    LongBenchV2OutputsTableRow,
)
from .predict import LongBenchV2Output


class LongBenchV2Runner(
    BaseBenchmarkRunner[
        LongBenchV2Output, LongBenchV2OutputsTableRow, LongBenchV2LeaderBoardTable
    ]
):
    def _make_leaderboard_table(
        self, outputs: list[LongBenchV2OutputsTableRow]
    ) -> LongBenchV2LeaderBoardTable: ...

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        batchsize: int,
    ) -> list[LongBenchV2Output]: ...

    def _to_output_row(
        self, *, task: str, subtask: str, output: LongBenchV2Output
    ) -> LongBenchV2OutputsTableRow: ...

    def _to_outputs_table(
        self, name: str, rows: list[LongBenchV2OutputsTableRow]
    ) -> OutputsTable[LongBenchV2OutputsTableRow]: ...
