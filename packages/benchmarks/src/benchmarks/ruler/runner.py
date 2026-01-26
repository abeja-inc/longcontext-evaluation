from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from llm_inference.base import BaseGenerator
from transformers import AutoTokenizer

from .._base_benchmark.config import RunOptions
from .._base_benchmark.result import push_results_to_wandb
from .._base_benchmark.runner import BenchmarkRunner
from ..utils import filter_names, parse_csv_list
from .evaluation.config import TaskSetting
from .evaluation.evaluator import RulerEvaluator
from .prediction.predict import RulerPredictJob
from .evaluation.results import RulerResultsBuilder


def _load_task_settings(
    raw_tasks: list[dict[str, Any]],
    *,
    only_tasks: list[str],
    exclude_tasks: list[str],
    only_subsets: list[str],
    exclude_subsets: list[str],
) -> list[TaskSetting]:
    tasks: list[TaskSetting] = []
    parsed_tasks = [TaskSetting(**task) for task in raw_tasks]
    task_names = filter_names(
        [task.task for task in parsed_tasks], include=only_tasks, exclude=exclude_tasks
    )
    for task in parsed_tasks:
        if task.task not in task_names:
            continue
        filtered_filenames = filter_names(
            task.filenames,
            include=only_subsets,
            exclude=exclude_subsets,
            allow_stem=True,
        )
        tasks.append(task.model_copy(update={"filenames": filtered_filenames}))
    return tasks


def run_ruler(
    *,
    cfg: dict[str, Any],
    generator: BaseGenerator,
    generate_kwargs: dict[str, Any],
    model_name: str,
    model_root: Path,
    dataset_dir: Path,
    prediction_dir: Path,
    batchsize: int,
    logger: logging.Logger,
    wandb_run_factory: Callable[[dict[str, Any], str], Any],
) -> None:
    if not cfg.get("enabled", True):
        return

    tasks_path = Path(cfg["tasks_path"]).expanduser()
    with tasks_path.open("r", encoding="utf-8") as f:
        tasks_map: dict[str, list[str]] = yaml.safe_load(f)["tasks"]

    include_tasks = parse_csv_list(cfg.get("only_tasks"))
    exclude_tasks = parse_csv_list(cfg.get("exclude_tasks"))
    include_subsets = parse_csv_list(cfg.get("only_subsets"))
    exclude_subsets = parse_csv_list(cfg.get("exclude_subsets"))

    tokenizer = AutoTokenizer.from_pretrained(str(model_root / model_name))
    jobs: list[RulerPredictJob] = []
    task_names = filter_names(
        tasks_map.keys(), include=include_tasks, exclude=exclude_tasks
    )
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

    eval_config_path = Path(cfg["eval_config"]).expanduser()
    with eval_config_path.open("r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    tasks = _load_task_settings(
        eval_config.get("tasks", []),
        only_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
        only_subsets=include_subsets,
        exclude_subsets=exclude_subsets,
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
