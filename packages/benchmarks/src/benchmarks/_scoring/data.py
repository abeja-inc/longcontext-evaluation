from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class Score(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    benchmark: str
    language: str
    task: str
    subset: str
    index: int | str | None = None
    score: float
    context_length: int


@dataclass(frozen=True)
class Table:
    name: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Results:
    tables: list[Table]
    summary: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
