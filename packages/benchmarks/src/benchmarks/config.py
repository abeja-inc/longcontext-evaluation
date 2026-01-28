from pathlib import Path
from typing import Any, Literal

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
    output_root: Path
    tasks: list[TaskConfig]
