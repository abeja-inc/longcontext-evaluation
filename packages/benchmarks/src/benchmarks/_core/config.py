from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel

class SubtaskConfig(BaseModel):
    name: str
    language: Literal["english", "japanese"]
    dataset_filepath: Path
    output_filepath: Path
    metric: str
    settings: dict[str, Any] | None = None

class TaskConfig(BaseModel):
    name: str
    subtasks: list[SubtaskConfig]

class BenchmarkConfig(BaseModel):
    name: str
    tasks: list[TaskConfig]
