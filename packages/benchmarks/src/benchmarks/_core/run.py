from logging import Logger
from typing import Any

from llm_inference.base import BaseGenerator

from .config import BenchmarkConfig


def run_benchmarks(
    benchmark_configs: list[BenchmarkConfig],
    generator: BaseGenerator,
    generation_kwargs: dict[str, Any],
    logger: Logger,
) -> None:
    logger.info("Running benchmarks...")
    for config in benchmark_configs:
        logger.info(f"Running benchmark: {config.name}")
        benchmark_runner = get_runner(config.name)
        benchmark_runner.run(
            config=config,
            generator=generator,
            generation_kwargs=generation_kwargs,
            logger=logger,
        )
