from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar


GroupBy = Literal["subtask", "task", "language"]


@dataclass(frozen=True)
class BaseTableRow:
    model_name: str


TableRowType = TypeVar("TableRowType", bound=BaseTableRow)


@dataclass(frozen=True)
class BaseTable(Generic[TableRowType]):
    name: str
    rows: list[TableRowType]


TableType = TypeVar("TableType", bound=BaseTable)


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


OutputsTableRowType = TypeVar("OutputsTableRowType", bound=OutputsTableRow)


@dataclass(frozen=True)
class OutputsTable(BaseTable[OutputsTableRow]):
    pass


OutputsTableType = TypeVar("OutputsTableType", bound=OutputsTable)


@dataclass(frozen=True)
class MeanScoreTableRow(BaseTableRow):
    group: str
    mean_score: float


@dataclass(frozen=True)
class MeanScoreTable(BaseTable[MeanScoreTableRow]):
    pass


@dataclass(frozen=True)
class BenchmarkResults(Generic[OutputsTableRowType, OutputsTableType, TableType]):
    outputs_table: OutputsTableType
    mean_score_by_subtask: MeanScoreTable
    mean_score_by_task: MeanScoreTable
    mean_score_by_language: MeanScoreTable
    leaderboard_table: TableType

    @property
    def tables(self) -> list[BaseTable]:
        # 保存/ログ用に「全部欲しい」ならここでまとめて BaseTable に落とす
        return [
            self.outputs_table,
            self.mean_score_by_subtask,
            self.mean_score_by_task,
            self.mean_score_by_language,
            self.leaderboard_table,
        ]
