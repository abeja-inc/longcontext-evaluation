import json
from logging import Logger
from pathlib import Path

from ....utils import get_custom_logger
from .config import EvaluationConfig
from .data_model import EvaluationResult, Score, SubsetResult, TaskResult
from .scoring import MrcrScorer


class EvaluationPipeline:
    def __init__(
        self,
        output_filepath: Path,
        logger: Logger | None = None,
    ):
        self.output_filepath = output_filepath
        self.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.logger: Logger | None = logger if logger else get_custom_logger()

    def run(self, config: EvaluationConfig):
        results: list[TaskResult] = []
        for task_setting in config.tasks:
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

                score = self._evaluate(
                    pred_filepath=pred_filepath, metric=task_setting.metric
                )
                subset_results.append(
                    SubsetResult(subset_name=pred_filepath.stem, score=score)
                )
            results.append(
                TaskResult(
                    task=task_setting.task,
                    subsets=subset_results,
                )
            )
        eval_result = EvaluationResult(results=results)
        self.output_filepath.write_text(eval_result.model_dump_json(indent=4))
        self.logger.info("Evaluation completed.")

    def _evaluate(self, pred_filepath: Path, metric: str | None = None) -> list[Score]:
        with pred_filepath.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        scorer = MrcrScorer()
        return scorer.score_records(data)
