from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Sequence, TypeVar


GroupBy = Literal["subtask", "task", "language"]


@dataclass(frozen=True)
class BaseTableRow:
    model_name: str


TableRowType = TypeVar("TableRowType", bound=BaseTableRow, covariant=True)


@dataclass(frozen=True)
class BaseTable(Generic[TableRowType]):
    name: str
    rows: Sequence[TableRowType]


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
    output_reasoning: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


OutputsTableRowType = TypeVar(
    "OutputsTableRowType", bound=OutputsTableRow, covariant=True
)


@dataclass(frozen=True)
class OutputsTable(BaseTable[OutputsTableRowType], Generic[OutputsTableRowType]):
    pass


@dataclass(frozen=True)
class MeanScoreTableRow(BaseTableRow):
    group: str
    mean_score: float


@dataclass(frozen=True)
class MeanScoreTable(BaseTable[MeanScoreTableRow]):
    pass


LeaderboardTableType = TypeVar(
    "LeaderboardTableType", bound=BaseTable[Any], covariant=True
)


@dataclass(frozen=True)
class BenchmarkResults(Generic[OutputsTableRowType, LeaderboardTableType]):
    outputs_table: OutputsTable[OutputsTableRowType]
    mean_score_by_subtask: MeanScoreTable
    mean_score_by_task: MeanScoreTable
    mean_score_by_language: MeanScoreTable
    leaderboard_table: LeaderboardTableType

    @property
    def tables(self) -> list[BaseTable[Any]]:
        return [
            self.outputs_table,
            self.mean_score_by_subtask,
            self.mean_score_by_task,
            self.mean_score_by_language,
            self.leaderboard_table,
        ]
