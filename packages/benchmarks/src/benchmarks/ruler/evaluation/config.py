from pathlib import Path

from pydantic import BaseModel


class TaskSetting(BaseModel):
    task: str
    metric: str
    filenames: list[str]


class EvaluationConfig(BaseModel):
    prediction_dirpath: Path
    tasks: list[TaskSetting]
