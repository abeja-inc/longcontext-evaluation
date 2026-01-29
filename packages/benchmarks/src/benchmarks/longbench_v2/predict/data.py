from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from ..._core.predict import Output


DIFFICULTY = Literal["hard", "easy"]
LENGTH = Literal["short", "medium", "long"]
DOMAIN = Literal[
    "Single-Document QA",
    "Multi-Document QA",
    "Long In-context Learning",
    "Code Repository Understanding",
    "Long-dialogue History Understanding",
    "Long Structured Data Understanding",
]
SUBDOMAIN = Literal[
    "Academic",
    "Code repo QA",
    "Governmental",
    "User guide QA",
    "Financial",
    "Legal",
    "Literary",
    "Multi-news",
    "Detective",
    "Many-shot learning",
    "New language translation",
    "Event ordering",
    "Agent history QA",
    "Dialogue history QA",
    "Table QA",
    "Knowledge graph reasoning",
]


class RetrievedChunk(BaseModel):
    content: str
    c_idx: int
    score: float
    source: str


class LongBenchV2Input(BaseModel):
    id: str
    tokens: int
    question: str
    choice_A: str
    choice_B: str
    choice_C: str
    choice_D: str
    answer: Literal["A", "B", "C", "D"]
    context: str
    difficulty: DIFFICULTY
    length: LENGTH
    domain: DOMAIN
    sub_domain: SUBDOMAIN
    retrieved_context: list[RetrievedChunk] | None = None


@dataclass(frozen=True)
class LongBenchV2Output(Output):
    difficulty: DIFFICULTY
    length: LENGTH
    domain: DOMAIN
    sub_domain: SUBDOMAIN
