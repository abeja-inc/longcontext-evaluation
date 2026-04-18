from logging import Logger
from typing import Final

from ._core import Runner, RunnerConstructor
from ._core.evaluate import Bin
from .context_poisoning_resistance import ContextPoisoningResistanceRunner
from .longbench_v2 import LongBenchV2Runner
from .mrcr import OpenAIMRCRRunner
from .ruler import RULERRunner


RUNNER_REGISTRY: Final[dict[str, RunnerConstructor]] = {
    "context_poisoning_resistance": ContextPoisoningResistanceRunner,
    "longbench_v2": LongBenchV2Runner,
    "openai_mrcr": OpenAIMRCRRunner,
    "ruler": RULERRunner,
}


def get_runner(name: str, *, logger: Logger, bins: list[Bin] | None = None) -> Runner:
    logger.info(f"Initializing runner of type '{name}'")
    if name not in RUNNER_REGISTRY:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNER_REGISTRY)}")
    return RUNNER_REGISTRY[name](logger=logger, bins=bins)
