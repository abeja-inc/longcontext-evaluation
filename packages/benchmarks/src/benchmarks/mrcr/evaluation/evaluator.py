from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..._base_benchmark.interfaces import Evaluator
from . import EvaluationConfig, EvaluationPipeline
from .config import TaskSetting


class MRCREvaluator(Evaluator):
    name = "mrcr"

    def __init__(self, tasks: list[TaskSetting]) -> None:
        self._tasks = tasks

    def run(self, *, prediction_dir: Path, output_path: Path) -> dict[str, Any] | None:
        config = EvaluationConfig(prediction_dirpath=prediction_dir, tasks=self._tasks)
        pipeline = EvaluationPipeline(output_filepath=output_path)
        pipeline.run(config)

        if not output_path.exists():
            return None

        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)
