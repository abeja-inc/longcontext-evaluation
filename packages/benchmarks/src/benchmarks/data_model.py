from typing import Any

from pydantic import BaseModel


class Score(BaseModel):
    benchmark: str
    language: str
    task: str
    subset: str
    index: int | str | None = None
    score: float
    context_length: int
    tags: dict[str, Any] | None = None
