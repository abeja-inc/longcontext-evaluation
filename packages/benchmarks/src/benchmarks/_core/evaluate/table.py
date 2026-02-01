from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


@dataclass(frozen=True)
class BaseTableRow:
    model_name: str


TableRowType = TypeVar("TableRowType", bound=BaseTableRow, covariant=True)


@dataclass(frozen=True)
class BaseTable(Generic[TableRowType]):
    name: str
    rows: Sequence[TableRowType]


TableRowType = TypeVar("TableRowType", bound=BaseTableRow, covariant=True)


@dataclass(frozen=True)
class OutputsTableRow(BaseTableRow):
    task: str
    subtask: str
    language: str
    context_length: int

    input: str
    answer: str
    score: float
    output: str
    output_reasoning: str | None


OutputsTableRowType = TypeVar(
    "OutputsTableRowType", bound=OutputsTableRow, covariant=True
)


@dataclass(frozen=True)
class OutputsTable(BaseTable[OutputsTableRowType], Generic[OutputsTableRowType]):
    pass


@dataclass(frozen=True)
class MeanScoreByLengthTableRow(BaseTableRow):
    group: str
    context_length: str
    mean_score: float


@dataclass(frozen=True)
class MeanScoreByLengthTable(BaseTable[MeanScoreByLengthTableRow]): ...


LeaderboardTableRowType = TypeVar(
    "LeaderboardTableRowType", bound=BaseTableRow, covariant=True
)


@dataclass(frozen=True)
class BenchmarkResults(Generic[OutputsTableRowType, LeaderboardTableRowType]):
    outputs_table: OutputsTable[OutputsTableRowType]
    mean_score_by_subtask: MeanScoreByLengthTable
    mean_score_by_task: MeanScoreByLengthTable
    mean_score_by_language: MeanScoreByLengthTable
    leaderboard_table: BaseTable[LeaderboardTableRowType]
    additional_tables: Sequence[BaseTable[BaseTableRow]]

    @property
    def tables(self) -> Sequence[BaseTable[BaseTableRow]]:
        return (
            self.outputs_table,
            self.mean_score_by_subtask,
            self.mean_score_by_task,
            self.mean_score_by_language,
            self.leaderboard_table,
            *self.additional_tables,
        )
