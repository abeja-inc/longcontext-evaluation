import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from llm_inference.base import BaseGenerator
from transformers import AutoTokenizer

from .._core.config import RunOptions
from .._core.runner import BenchmarkRunner
from .._scoring.push_wandb import push_results_to_wandb
from .evaluation.config import TaskSetting
from .evaluation.evaluator import RulerEvaluator
from .evaluation.results_builder import RulerResultsBuilder
from .prediction.predictor import RulerPredictJob


def _load_task_settings(task_configs: list[dict[str, Any]]) -> list[TaskSetting]:
    return [
        TaskSetting(
            task=task["task"],
            metric=task["metric"],
            filenames=list(task["filenames"]),
        )
        for task in raw_tasks
    ]


def run_ruler(
    *,
    cfg: dict[str, Any],
    generator: BaseGenerator,
    generate_kwargs: dict[str, Any],
    tokenizer: AutoTokenizer,
    dataset_dir: Path,
    prediction_dir: Path,
    batchsize: int,
    logger: logging.Logger,
    wandb_run_factory: Callable[[dict[str, Any], str], Any],
) -> None:
    if not cfg.get("enabled", True):
        return

    config_dir = Path(cfg.get("_config_dir", "."))
    tasks_config = cfg.get("tasks")
    if isinstance(tasks_config, dict):
        tasks_map: dict[str, list[str]] = tasks_config["tasks"]
    else:
        tasks_path = Path(cfg["tasks_path"]).expanduser()
        if not tasks_path.is_absolute():
            tasks_path = config_dir / tasks_path
        with tasks_path.open("r", encoding="utf-8") as f:
            tasks_map = yaml.safe_load(f)["tasks"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_root / model_name))
    jobs: list[RulerPredictJob] = []

    for task_name in task_names:
        dataset_filenames = tasks_map.get(task_name, [])
        filtered_filenames = filter_names(
            dataset_filenames,
            include=include_subsets,
            exclude=exclude_subsets,
            allow_stem=True,
        )
        for dataset_filename in filtered_filenames:
            pred_filepath = prediction_dir / task_name / dataset_filename
            dataset_filepath = dataset_dir / task_name / dataset_filename
            jobs.append(
                RulerPredictJob(
                    name=f"{task_name}/{dataset_filename}",
                    dataset_path=dataset_filepath,
                    pred_path=pred_filepath,
                    tokenizer=tokenizer,
                )
            )

    if cfg.get("run_pred", True):
        runner = BenchmarkRunner(
            generator=generator, generate_kwargs=generate_kwargs, logger=logger
        )
        runner.run(
            jobs=jobs,
            model_name=model_name,
            opts=RunOptions(batch_size=batchsize, prediction_dir=prediction_dir),
        )

    if not cfg.get("run_eval", True):
        return

    eval_config = cfg.get("eval_config")
    if isinstance(eval_config, dict):
        eval_config = eval_config
    else:
        eval_config_path = Path(
            cfg.get("eval_config_path") or cfg.get("eval_config")
        ).expanduser()
        if not eval_config_path.is_absolute():
            eval_config_path = config_dir / eval_config_path
        with eval_config_path.open("r", encoding="utf-8") as f:
            eval_config = yaml.safe_load(f)

    tasks = _load_task_settings(
        eval_config.get("tasks", []),
    )

    evaluator = RulerEvaluator(tasks=tasks)
    output_filepath = prediction_dir / "summary.json"
    summary_json = evaluator.run(
        prediction_dir=prediction_dir, output_path=output_filepath
    )

    results_builder = RulerResultsBuilder()
    results = results_builder.build_results(
        model_name=model_name,
        prediction_dir=prediction_dir,
        summary_json=summary_json,
    )

    run = wandb_run_factory(cfg.get("wandb", {}), "RULER")
    if run and push_results_to_wandb is not None:
        push_results_to_wandb(results=results, wandb_run=run)
    if run is not None:
        run.finish()
