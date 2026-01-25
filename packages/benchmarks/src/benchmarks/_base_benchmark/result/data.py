from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Table:
    name: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Results:
    tables: list[Table]
    summary: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
