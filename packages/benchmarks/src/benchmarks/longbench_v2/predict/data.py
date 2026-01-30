from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from ..._core.predict.data import Input, Output


class Difficulty(StrEnum):
    EASY = "easy"
    HARD = "hard"


class Length(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Domain(StrEnum):
    SINGLE_DOCUMENT_QA = "Single-Document QA"
    MULTI_DOCUMENT_QA = "Multi-Document QA"
    LONG_IN_CONTEXT_LEARNING = "Long In-context Learning"
    CODE_REPOSITORY_UNDERSTANDING = "Code Repository Understanding"
    LONG_DIALOGUE_HISTORY_UNDERSTANDING = "Long-dialogue History Understanding"
    LONG_STRUCTURED_DATA_UNDERSTANDING = "Long Structured Data Understanding"


class SubDomain(StrEnum):
    ACADEMIC = "Academic"
    CODE_REPO_QA = "Code repo QA"
    GOVERNMENTAL = "Governmental"
    USER_GUIDE_QA = "User guide QA"
    FINANCIAL = "Financial"
    LEGAL = "Legal"
    LITERARY = "Literary"
    MULTI_NEWS = "Multi-news"
    DETECTIVE = "Detective"
    MANY_SHOT_LEARNING = "Many-shot learning"
    NEW_LANGUAGE_TRANSLATION = "New language translation"
    EVENT_ORDERING = "Event ordering"
    AGENT_HISTORY_QA = "Agent history QA"
    DIALOGUE_HISTORY_QA = "Dialogue history QA"
    TABLE_QA = "Table QA"
    KNOWLEDGE_GRAPH_REASONING = "Knowledge graph reasoning"


class RetrievedChunk(BaseModel):
    content: str
    c_idx: int
    score: float
    source: str


class LongBenchV2Input(Input):
    question: str
    choice_A: str
    choice_B: str
    choice_C: str
    choice_D: str
    answer: Literal["A", "B", "C", "D"]
    context: str
    difficulty: Difficulty
    length: Length
    domain: Domain
    sub_domain: SubDomain
    retrieved_context: list[RetrievedChunk] | None = None


class LongBenchV2Output(Output):
    difficulty: Difficulty
    length: Length
    domain: Domain
    sub_domain: SubDomain
