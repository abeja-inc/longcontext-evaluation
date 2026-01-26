import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator
from vllm import SamplingParams

from benchmarks._base_benchmark.config import RunOptions
from benchmarks._base_benchmark.runner import BenchmarkRunner
from benchmarks.mrcr.predict import MRCRPredictJob
from benchmarks.utils import filter_names, get_custom_logger, parse_csv_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True, help="Path to models root directory")
    p.add_argument("--model-name", type=str, required=True, help="Name of the model")
    p.add_argument(
        "--vllm-config",
        type=Path,
        default=Path("./vllm_offline_config.yml"),
        help="Path to vllm config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/longcontext-llm-benchmark-eval/experimentation/outputs"),
        help="Path to output directory",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("~/datasets"), help="Path to dataset directory"
    )
    p.add_argument("--tasks", type=Path, default=Path("./tasks.yml"), help="Path to tasks file")
    p.add_argument("--batchsize", type=int, default=1, help="Batch size for inference")
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


def build_generator(
    *, model_path: Path, vllm_config_path: Path, logger: logging.Logger
) -> tuple[VLLMOfflineGenerator, dict[str, Any]]:
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

    generator_kwargs = {
        k: v
        for k, v in serve_cfg.items()
        if k not in {"extra_args", "model_name_or_path", "max_model_len"}
    }

    generator = VLLMOfflineGenerator(
        model_name=str(model_path),
        max_context_length=max_model_len,
        max_output_tokens=max_new_tokens,
        logger=logger,
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

    return generator, generate_kwargs


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    model_path: Path = args.model_root.expanduser() / args.model_name
    dataset_dirpath: Path = args.dataset_dir.expanduser()
    output_dirpath: Path = args.output_dir.expanduser() / args.model_name

    with args.tasks.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        tasks: dict[str, list[str]] = raw["tasks"]

    generator, generate_kwargs = build_generator(
        model_path=model_path, vllm_config_path=args.vllm_config, logger=logger
    )

    include_tasks = parse_csv_list(args.only_tasks)
    exclude_tasks = parse_csv_list(args.exclude_tasks)
    include_subsets = parse_csv_list(args.only_subsets)
    exclude_subsets = parse_csv_list(args.exclude_subsets)

    jobs: list[MRCRPredictJob] = []
    task_names = filter_names(tasks.keys(), include=include_tasks, exclude=exclude_tasks)
    for task_name in task_names:
        dataset_filenames = tasks.get(task_name, [])
        filtered_filenames = filter_names(
            dataset_filenames,
            include=include_subsets,
            exclude=exclude_subsets,
            allow_stem=True,
        )
        for dataset_filename in filtered_filenames:
            pred_filepath = output_dirpath / task_name / dataset_filename
            dataset_filepath = dataset_dirpath / task_name / dataset_filename
            jobs.append(
                MRCRPredictJob(
                    name=f"{task_name}/{dataset_filename}",
                    dataset_path=dataset_filepath,
                    pred_path=pred_filepath,
                )
            )

    if not jobs:
        logger.warning("No tasks found in %s", args.tasks)
        return

    runner = BenchmarkRunner(
        generator=generator, generate_kwargs=generate_kwargs, logger=logger
    )
    runner.run(
        jobs=jobs,
        model_name=args.model_name,
        opts=RunOptions(batch_size=args.batchsize, prediction_dir=output_dirpath),
    )


if __name__ == "__main__":
    args = parse_args()
    logger = get_custom_logger()
    main(args=args, logger=logger)
