import argparse
import logging
from pathlib import Path
from typing import Any

import wandb
import yaml
from benchmark import BenchmarkConfig, SubtaskConfig, TaskConfig, run_benchmarks
from llm_inference import get_generator
from openai import OpenAI


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 二重に handler が付くのを防ぐ
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./run_configs/openai_api.yml"),
        help="Path to benchmark run config",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load base configuration
    base_config_path = config.get("base_config")
    if base_config_path:
        base_config_path = Path(base_config_path)
        with base_config_path.open("r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        config = merge_dicts(base_config, config)
    return config


def parse_benchmark_configs(
    config: dict[str, Any], dataset_root: Path, output_root: Path
) -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            name=benchmark_name,
            output_root=output_root / benchmark_name,
            tasks=[
                TaskConfig(
                    name=task_name,
                    subtasks=[
                        SubtaskConfig(
                            name=subtask_name,
                            language=subtask_config["language"],
                            dataset_filepath=dataset_root
                            / benchmark_name
                            / subtask_config["dataset"],
                            output_filepath=output_root
                            / benchmark_name
                            / subtask_config["output"],
                            metric=subtask_config["metric"],
                            settings=subtask_config["settings"],
                        )
                        for subtask_name, subtask_config in task_config.items()
                    ],
                )
                for task_name, task_config in benchmark_config.items()
            ],
        )
        for benchmark_name, benchmark_config in config.items()
    ]


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    logger = get_logger(name="run-benchmarks", level=logging.INFO)

    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize wandb
    wandb_config = config.get("wandb")
    if wandb_config:
        wandb.init(**wandb_config)
        log_wandb = True
    else:
        log_wandb = False

    # Initialize dataset and output paths
    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_root"])

    # Parse benchmark configurations
    batchsize = int(config["batchsize"])
    benchmark_configs = parse_benchmark_configs(
        config=config,
        dataset_root=dataset_root,
        output_root=output_root,
    )

    # Initialize generator
    generator_config = config["generator"]
    generator_type = generator_config.pop("type")
    if "client" in generator_config:
        client = OpenAI(**generator_config["client"])
        generator = get_generator(
            type=generator_type, client=client, logger=logger, **generator_config
        )
    else:
        generator = get_generator(
            type=generator_type, logger=logger, **generator_config
        )

    # Run evaluation
    run_benchmarks(
        generator=generator,
        generation_kwargs=config["generation_kwargs"],
        batchsize=batchsize,
        benchmark_configs=benchmark_configs,
        logger=logger,
        log_wandb=log_wandb,
    )

    # Finish wandb
    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
