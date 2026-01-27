import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._core.interfaces import Evaluator
from ..._core.utils import read_jsonl
from .scorer import MrcrScorer


@dataclass(frozen=True)
class TaskSetting:
    task: str
    metric: str
    filenames: list[str]


class MRCREvaluator(Evaluator):
    name = "mrcr"

    def __init__(self, tasks: list[TaskSetting]) -> None:
        self._tasks = tasks

    def run(self, *, prediction_dir: Path, output_path: Path) -> dict[str, Any] | None:
        results: list[dict[str, Any]] = []
        scorer = MrcrScorer()

        for task_setting in self._tasks:
            subset_results: list[dict[str, Any]] = []
            for pred_filename in task_setting.filenames:
                pred_filepath = prediction_dir / task_setting.task / pred_filename
                if not pred_filepath.is_file():
                    continue

                records = read_jsonl(pred_filepath)
                scores = scorer.score_records(
                    records,
                    task=task_setting.task,
                    subset=pred_filepath.stem,
                )
                subset_results.append(
                    {
                        "subset_name": pred_filepath.stem,
                        "score": [score.model_dump() for score in scores],
                    }
                )

            results.append({"task": task_setting.task, "subsets": subset_results})

        summary: dict[str, Any] = {"results": results}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        return summary


__all__ = ["MRCREvaluator", "TaskSetting"]
