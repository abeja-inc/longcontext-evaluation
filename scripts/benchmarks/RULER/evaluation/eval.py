import argparse
from pathlib import Path

import wandb
import yaml

from benchmarks._base_benchmark.result import push_results_to_wandb
from benchmarks.ruler.evaluation.config import TaskSetting
from benchmarks.ruler.evaluator import RulerEvaluator
from benchmarks.ruler.results import RulerResultsBuilder
from benchmarks.utils import filter_names, parse_csv_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path("./eval_config.yml"), help="Path to YAML config file"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/longcontext-llm-benchmark-eval/experimentation/outputs"),
        help="Path to output directory",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--group", type=str, default="RULER")
    p.add_argument("--model-name", type=str, required=True, help="model name")
    p.add_argument(
        "--only-tasks",
        type=str,
        default=None,
        help="Comma-separated task name patterns to include",
    )
    p.add_argument(
        "--exclude-tasks",
        type=str,
        default=None,
        help="Comma-separated task name patterns to exclude",
    )
    p.add_argument(
        "--only-subsets",
        type=str,
        default=None,
        help="Comma-separated subset filename patterns to include",
    )
    p.add_argument(
        "--exclude-subsets",
        type=str,
        default=None,
        help="Comma-separated subset filename patterns to exclude",
    )
    return p.parse_args()


def main(args: argparse.Namespace):
    with args.config.open("r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    include_tasks = parse_csv_list(args.only_tasks)
    exclude_tasks = parse_csv_list(args.exclude_tasks)
    include_subsets = parse_csv_list(args.only_subsets)
    exclude_subsets = parse_csv_list(args.exclude_subsets)

    tasks: list[TaskSetting] = []
    raw_tasks = [TaskSetting(**task) for task in eval_config.get("tasks", [])]
    task_names = filter_names(
        [task.task for task in raw_tasks], include=include_tasks, exclude=exclude_tasks
    )
    for task in raw_tasks:
        if task.task not in task_names:
            continue
        filtered_filenames = filter_names(
            task.filenames,
            include=include_subsets,
            exclude=exclude_subsets,
            allow_stem=True,
        )
        tasks.append(task.model_copy(update={"filenames": filtered_filenames}))
    output_dirpath = args.output_dir.expanduser()
    model_name = args.model_name

    prediction_dirpath = output_dirpath / model_name
    output_filepath = prediction_dirpath / "summary.json"

    evaluator = RulerEvaluator(tasks=tasks)
    summary_json = evaluator.run(
        prediction_dir=prediction_dirpath, output_path=output_filepath
    )

    results_builder = RulerResultsBuilder()
    results = results_builder.build_results(
        model_name=model_name,
        prediction_dir=prediction_dirpath,
        summary_json=summary_json,
    )

    if args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or model_name,
            group=args.group,
            config={
                "model_name": model_name,
                "output_dir": str(output_dirpath),
                "config_path": str(args.config),
            },
        )

        if push_results_to_wandb is None:
            raise RuntimeError("wandb is not available for logging results.")
        push_results_to_wandb(results=results, wandb_run=run)

        artifact_name = f"{model_name}-ruler-eval".replace("/", "_")
        art = wandb.Artifact(
            name=artifact_name,
            type="evaluation",
            metadata={"model": model_name, "kind": "RULER"},
        )
        if output_filepath.exists():
            art.add_file(str(output_filepath))
            wandb.log_artifact(art)

        pred_dir = prediction_dirpath
        if pred_dir.exists():
            jsonl_files = sorted(pred_dir.rglob("*.jsonl"))
            if jsonl_files:
                artifact_name = f"{model_name}-ruler-preds".replace("/", "_")
                preds_art = wandb.Artifact(
                    name=artifact_name,
                    type="predictions",
                    metadata={"model": model_name, "kind": "RULER", "count": len(jsonl_files)},
                )
                for fp in jsonl_files:
                    preds_art.add_file(str(fp), name=str(fp.relative_to(pred_dir)))
                wandb.log_artifact(preds_art)

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
