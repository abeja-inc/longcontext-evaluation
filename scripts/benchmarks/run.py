import argparse
import logging
from pathlib import Path
from typing import Any

import wandb
import yaml
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator
from transformers import AutoTokenizer
from vllm import SamplingParams

from benchmarks._base_benchmark.config import RunOptions
from benchmarks._base_benchmark.result import push_results_to_wandb
from benchmarks._base_benchmark.runner import BenchmarkRunner
from benchmarks.longbench_v2.predict import build_jobs_for_dataset_dir, load_prompt_templates
from benchmarks.longbench_v2.results import LongBenchResultsBuilder
from benchmarks.mrcr.evaluation.config import TaskSetting as MRCRTaskSetting
from benchmarks.mrcr.evaluator import MRCREvaluator
from benchmarks.mrcr.predict import MRCRPredictJob
from benchmarks.mrcr.results import MRCRResultsBuilder
from benchmarks.ruler.evaluation.config import TaskSetting as RulerTaskSetting
from benchmarks.ruler.evaluator import RulerEvaluator
from benchmarks.ruler.predict import RulerPredictJob
from benchmarks.ruler.results import RulerResultsBuilder
from benchmarks.utils import filter_names, get_custom_logger, parse_csv_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("./run_config.yml"),
        help="Path to benchmark run config",
    )
    p.add_argument(
        "--only-benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names to include",
    )
    p.add_argument(
        "--exclude-benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names to exclude",
    )
    return p.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def build_generator(
    *, model_path: Path, vllm_config_path: Path, logger: logging.Logger
) -> tuple[VLLMOfflineGenerator, dict[str, Any], int, int]:
    with vllm_config_path.open("r", encoding="utf-8") as f:
        vllm_config = yaml.safe_load(f)

    serve_cfg: dict[str, Any] = vllm_config.get("serve", {})
    generation_cfg: dict[str, Any] = vllm_config.get("generation", {})

    extra_args = serve_cfg.get("extra_args", {})
    max_model_len = extra_args.get("max_model_len", serve_cfg.get("max_model_len", 4096))
    max_new_tokens = generation_cfg.get(
        "max_new_tokens",
        generation_cfg.get("max_output_tokens", generation_cfg.get("max_tokens", 512)),
    )

    reasoning_parser = serve_cfg.get("reasoning_parser")
    generator_kwargs = {
        k: v
        for k, v in serve_cfg.items()
        if k
        not in {
            "extra_args",
            "model_name_or_path",
            "max_model_len",
            "reasoning_parser",
        }
    }

    generator = VLLMOfflineGenerator(
        model_name=str(model_path),
        max_context_length=max_model_len,
        max_output_tokens=max_new_tokens,
        logger=logger,
        reasoning_parser=reasoning_parser,
        **generator_kwargs,
        **extra_args,
    )

    sampling_params = SamplingParams(
        **{
            k: v
            for k, v in generation_cfg.items()
            if k
            not in {
                "max_new_tokens",
                "max_output_tokens",
                "max_tokens",
                "chat_template_kwargs",
                "buffer_tokens",
            }
        }
    )

    generate_kwargs = {
        "sampling_params": sampling_params,
        "buffer_tokens": generation_cfg.get("buffer_tokens", 10),
        "chat_template_kwargs": generation_cfg.get("chat_template_kwargs", {}),
    }

    return generator, generate_kwargs, max_model_len, max_new_tokens


def _init_wandb(
    *,
    wandb_cfg: dict[str, Any],
    model_name: str,
    group: str,
) -> wandb.sdk.wandb_run.Run | None:
    project = wandb_cfg.get("project")
    if not project:
        return None
    return wandb.init(
        project=project,
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name") or model_name,
        group=wandb_cfg.get("group") or group,
        config=wandb_cfg.get("config", {}),
    )


def _finalize_wandb(run: wandb.sdk.wandb_run.Run | None) -> None:
    if run is not None:
        wandb.finish()


def run_longbench(
    *,
    cfg: dict[str, Any],
    model_name: str,
    model_root: Path,
    vllm_config: Path,
    output_dir: Path,
    dataset_dir: Path,
    batchsize: int,
    logger: logging.Logger,
) -> None:
    if not cfg.get("enabled", True):
        return

    generator, generate_kwargs, max_model_len, max_new_tokens = build_generator(
        model_path=model_root / model_name,
        vllm_config_path=vllm_config,
        logger=logger,
    )

    prediction_dir = output_dir / model_name
    prompt_templates = load_prompt_templates(expand_path(cfg["tasks_path"]))
    tokenizer = AutoTokenizer.from_pretrained(str(model_root / model_name))

    jobs = build_jobs_for_dataset_dir(
        dataset_dir=dataset_dir,
        prediction_dir=prediction_dir,
        prompt_templates=prompt_templates,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        rag_topn=cfg.get("rag_topn", 0),
        cot=cfg.get("cot", False),
        no_context=cfg.get("no_context", False),
        include_datasets=parse_csv_list(cfg.get("only_datasets")),
        exclude_datasets=parse_csv_list(cfg.get("exclude_datasets")),
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

    results_builder = LongBenchResultsBuilder(
        compensate_missing=cfg.get("compensate_missing", False)
    )
    results = results_builder.build_results(
        model_name=model_name, prediction_dir=prediction_dir, summary_json=None
    )

    run = _init_wandb(
        wandb_cfg=cfg.get("wandb", {}), model_name=model_name, group="LongBench-v2"
    )
    if run and push_results_to_wandb is not None:
        push_results_to_wandb(results=results, wandb_run=run)
    _finalize_wandb(run)


def _load_task_settings(
    raw_tasks: list[dict[str, Any]],
    *,
    task_cls: type[MRCRTaskSetting] | type[RulerTaskSetting],
    only_tasks: list[str],
    exclude_tasks: list[str],
    only_subsets: list[str],
    exclude_subsets: list[str],
) -> list[MRCRTaskSetting] | list[RulerTaskSetting]:
    tasks: list[MRCRTaskSetting] | list[RulerTaskSetting] = []
    parsed_tasks = [task_cls(**task) for task in raw_tasks]
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


def run_mrcr(
    *,
    cfg: dict[str, Any],
    model_name: str,
    model_root: Path,
    vllm_config: Path,
    output_dir: Path,
    dataset_dir: Path,
    batchsize: int,
    logger: logging.Logger,
) -> None:
    if not cfg.get("enabled", True):
        return

    generator, generate_kwargs, _, _ = build_generator(
        model_path=model_root / model_name,
        vllm_config_path=vllm_config,
        logger=logger,
    )

    prediction_dir = output_dir / model_name
    tasks_path = expand_path(cfg["tasks_path"])
    with tasks_path.open("r", encoding="utf-8") as f:
        raw_tasks: dict[str, list[str]] = yaml.safe_load(f)["tasks"]

    include_tasks = parse_csv_list(cfg.get("only_tasks"))
    exclude_tasks = parse_csv_list(cfg.get("exclude_tasks"))
    include_subsets = parse_csv_list(cfg.get("only_subsets"))
    exclude_subsets = parse_csv_list(cfg.get("exclude_subsets"))

    jobs: list[MRCRPredictJob] = []
    task_names = filter_names(raw_tasks.keys(), include=include_tasks, exclude=exclude_tasks)
    for task_name in task_names:
        dataset_filenames = raw_tasks.get(task_name, [])
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
                MRCRPredictJob(
                    name=f"{task_name}/{dataset_filename}",
                    dataset_path=dataset_filepath,
                    pred_path=pred_filepath,
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

    eval_config_path = expand_path(cfg["eval_config"])
    with eval_config_path.open("r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    tasks = _load_task_settings(
        eval_config.get("tasks", []),
        task_cls=MRCRTaskSetting,
        only_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
        only_subsets=include_subsets,
        exclude_subsets=exclude_subsets,
    )

    evaluator = MRCREvaluator(tasks=tasks)
    output_filepath = prediction_dir / "summary.json"
    summary_json = evaluator.run(
        prediction_dir=prediction_dir, output_path=output_filepath
    )

    results_builder = MRCRResultsBuilder()
    results = results_builder.build_results(
        model_name=model_name,
        prediction_dir=prediction_dir,
        summary_json=summary_json,
    )

    run = _init_wandb(wandb_cfg=cfg.get("wandb", {}), model_name=model_name, group="MRCR")
    if run and push_results_to_wandb is not None:
        push_results_to_wandb(results=results, wandb_run=run)
    _finalize_wandb(run)


def run_ruler(
    *,
    cfg: dict[str, Any],
    model_name: str,
    model_root: Path,
    vllm_config: Path,
    output_dir: Path,
    dataset_dir: Path,
    batchsize: int,
    logger: logging.Logger,
) -> None:
    if not cfg.get("enabled", True):
        return

    generator, generate_kwargs, _, _ = build_generator(
        model_path=model_root / model_name,
        vllm_config_path=vllm_config,
        logger=logger,
    )

    prediction_dir = output_dir / model_name
    tasks_path = expand_path(cfg["tasks_path"])
    with tasks_path.open("r", encoding="utf-8") as f:
        raw_tasks: dict[str, list[str]] = yaml.safe_load(f)["tasks"]

    include_tasks = parse_csv_list(cfg.get("only_tasks"))
    exclude_tasks = parse_csv_list(cfg.get("exclude_tasks"))
    include_subsets = parse_csv_list(cfg.get("only_subsets"))
    exclude_subsets = parse_csv_list(cfg.get("exclude_subsets"))

    tokenizer = AutoTokenizer.from_pretrained(str(model_root / model_name))
    jobs: list[RulerPredictJob] = []
    task_names = filter_names(raw_tasks.keys(), include=include_tasks, exclude=exclude_tasks)
    for task_name in task_names:
        dataset_filenames = raw_tasks.get(task_name, [])
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

    eval_config_path = expand_path(cfg["eval_config"])
    with eval_config_path.open("r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    tasks = _load_task_settings(
        eval_config.get("tasks", []),
        task_cls=RulerTaskSetting,
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

    run = _init_wandb(wandb_cfg=cfg.get("wandb", {}), model_name=model_name, group="RULER")
    if run and push_results_to_wandb is not None:
        push_results_to_wandb(results=results, wandb_run=run)
    _finalize_wandb(run)


def main() -> None:
    args = parse_args()
    config_path = expand_path(args.config)
    logger = get_custom_logger()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model_root = expand_path(model_cfg.get("root", "./models"))
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("model.name is required in config.")

    vllm_config = expand_path(config.get("vllm_config", "./vllm_offline_config.yml"))
    output_dir = expand_path(config.get("output_dir", "./outputs"))
    dataset_dir = expand_path(config.get("dataset_dir", "./datasets"))
    batchsize = int(config.get("batchsize", 1))

    only_benchmarks = parse_csv_list(args.only_benchmarks)
    exclude_benchmarks = parse_csv_list(args.exclude_benchmarks)

    benchmarks_cfg = config.get("benchmarks", {})
    benchmark_names = filter_names(
        benchmarks_cfg.keys(), include=only_benchmarks, exclude=exclude_benchmarks
    )

    for benchmark_name in benchmark_names:
        bench_cfg = benchmarks_cfg.get(benchmark_name, {})
        if benchmark_name == "longbench_v2":
            run_longbench(
                cfg=bench_cfg,
                model_name=model_name,
                model_root=model_root,
                vllm_config=vllm_config,
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                batchsize=batchsize,
                logger=logger,
            )
        elif benchmark_name == "mrcr":
            run_mrcr(
                cfg=bench_cfg,
                model_name=model_name,
                model_root=model_root,
                vllm_config=vllm_config,
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                batchsize=batchsize,
                logger=logger,
            )
        elif benchmark_name == "ruler":
            run_ruler(
                cfg=bench_cfg,
                model_name=model_name,
                model_root=model_root,
                vllm_config=vllm_config,
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                batchsize=batchsize,
                logger=logger,
            )
        else:
            logger.warning("Unknown benchmark: %s", benchmark_name)


if __name__ == "__main__":
    main()
