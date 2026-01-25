from pydantic import BaseModel

from ..data_model import Score


class SubsetResult(BaseModel):
    subset_name: str
    score: list[Score]


class TaskResult(BaseModel):
    task: str
    subsets: list[SubsetResult]


class EvaluationResult(BaseModel):
    results: list[TaskResult]
