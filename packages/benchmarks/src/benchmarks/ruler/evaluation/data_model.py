from pydantic import BaseModel


class Score(BaseModel):
    score: float
    context_length: int


class SubsetResult(BaseModel):
    subset_name: str
    score: list[Score]


class TaskResult(BaseModel):
    task: str
    subsets: list[SubsetResult]


class EvaluationResult(BaseModel):
    results: list[TaskResult]
