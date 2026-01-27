from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSetting:
    task: str
    metric: str
    filenames: list[str]
