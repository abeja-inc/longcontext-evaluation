from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..._core.interfaces import Evaluator
from ..._core.utils import read_jsonl
from ..._scoring import Score
from ....utils import get_custom_logger
from .scorer import build_scores


class TaskSetting(BaseModel):
    task: str
    metric: str = "default"
    filenames: list[str]


class EvaluationConfig(BaseModel):
    prediction_dirpath: Path
    tasks: list[TaskSetting] = Field(default_factory=list)
    compensate_missing: bool = False


class SubsetResult(BaseModel):
    subset_name: str
    score: list[Score]


class TaskResult(BaseModel):
    task: str
    subsets: list[SubsetResult]


class EvaluationResult(BaseModel):
    results: list[TaskResult]


class EvaluationPipeline:
    def __init__(self, output_filepath: Path, logger: Logger | None = None):
        self.output_filepath = output_filepath
        self.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger if logger else get_custom_logger()

    def run(self, config: EvaluationConfig) -> None:
        task_settings = config.tasks or _build_task_settings_from_dir(
            config.prediction_dirpath
        )
        results: list[TaskResult] = []
        for task_setting in task_settings:
            self.logger.info("Evaluating task '%s'...", task_setting.task)
            subset_results: list[SubsetResult] = []
            for pred_filename in task_setting.filenames:
                pred_filepath = (
                    config.prediction_dirpath / task_setting.task / pred_filename
                )
                if not pred_filepath.is_file():
                    self.logger.warning(
                        "Prediction file '%s' not found.", pred_filepath
                    )
                    continue
                scores = self._evaluate(
                    pred_filepath=pred_filepath,
                    task=task_setting.task,
                    subset=pred_filepath.stem,
                    compensate_missing=config.compensate_missing,
                )
                subset_results.append(
                    SubsetResult(subset_name=pred_filepath.stem, score=scores)
                )
            results.append(TaskResult(task=task_setting.task, subsets=subset_results))

        eval_result = EvaluationResult(results=results)
        self.output_filepath.write_text(eval_result.model_dump_json(indent=4))
        self.logger.info("Evaluation completed.")

    def _evaluate(
        self,
        pred_filepath: Path,
        *,
        task: str,
        subset: str,
        compensate_missing: bool,
    ) -> list[Score]:
        records = read_jsonl(pred_filepath)
        scores, _rows = build_scores(
            records,
            prompt_type=task,
            subset_name=subset,
            compensate_missing=compensate_missing,
        )
        return scores


class LongBenchEvaluator(Evaluator):
    name = "longbench_v2"

    def __init__(
        self,
        *,
        tasks: list[TaskSetting] | None = None,
        compensate_missing: bool = False,
    ) -> None:
        self._tasks = tasks or []
        self._compensate_missing = compensate_missing

    def run(self, *, prediction_dir: Path, output_path: Path) -> dict[str, Any] | None:
        config = EvaluationConfig(
            prediction_dirpath=prediction_dir,
            tasks=self._tasks,
            compensate_missing=self._compensate_missing,
        )
        pipeline = EvaluationPipeline(output_filepath=output_path)
        pipeline.run(config)

        if not output_path.exists():
            return None

        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)


def _build_task_settings_from_dir(prediction_dir: Path) -> list[TaskSetting]:
    task_map: dict[str, list[str]] = {}
    for filepath in sorted(prediction_dir.rglob("*.jsonl")):
        task_map.setdefault(filepath.parent.name, []).append(filepath.name)
    return [
        TaskSetting(task=task, filenames=sorted(filenames))
        for task, filenames in task_map.items()
    ]
