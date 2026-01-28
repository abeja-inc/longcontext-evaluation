from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class Output:
    input: str
    answer: object
    output: str
    context_length: int

    index: int | str | None = None
    output_reasoning: str | None = None


OutputType = TypeVar("OutputType", bound=Output)
