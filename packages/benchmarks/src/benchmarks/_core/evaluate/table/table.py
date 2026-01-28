from dataclasses import dataclass, field
from typing import Literal


GroupBy = Literal["subtask", "task", "language"]


@dataclass(frozen=True)
class BaseTableRow:
    model_name: str


@dataclass(frozen=True)
class BaseTable[T: BaseTableRow]:
    name: str
    rows: list[T]


class Output:
    input: str
    answer: object
    output: str
    context_length: int

    index: int | str | None = None
    output_reasoning: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


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


@dataclass(frozen=True)
class OutputsTable(BaseTable[OutputsTableRow]):
    pass


@dataclass(frozen=True)
class MeanScoreTableRow(BaseTableRow):
    group: str
    mean_score: float


@dataclass(frozen=True)
class MeanScoreTable(BaseTable[MeanScoreTableRow]):
    pass
