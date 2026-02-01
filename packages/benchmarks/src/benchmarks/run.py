from logging import Logger
from typing import Any

from llm_inference.base import BaseGenerator

from .config import BenchmarkConfig
from .runner_factory import get_runner


def run_benchmarks(
    generator: BaseGenerator,
    generation_kwargs: dict[str, Any],
    batchsize: int,
    benchmark_configs: list[BenchmarkConfig],
    logger: Logger,
    log_wandb: bool = False,
) -> None:
    logger.info("Running benchmarks...")
    for config in benchmark_configs:
        logger.info(f"Running benchmark: {config.name}")
        benchmark_runner = get_runner(
            name=config.name,
            logger=logger,
        )
        benchmark_runner.run(
            generator=generator,
            generation_kwargs=generation_kwargs,
            batchsize=batchsize,
            config=config,
            log_wandb=log_wandb,
        )
