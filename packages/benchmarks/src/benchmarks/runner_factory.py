from logging import Logger
from typing import Final

from ._core import Runner, RunnerConstructor
from ._core.evaluate import Bin
from .longbench_v2 import LongBenchV2Runner


RUNNER_REGISTRY: Final[dict[str, RunnerConstructor]] = {
    "longbench_v2": LongBenchV2Runner,
}


def get_runner(name: str, *, logger: Logger, bins: list[Bin] | None = None) -> Runner:
    if name not in RUNNER_REGISTRY:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNER_REGISTRY)}")
    return RUNNER_REGISTRY[name](logger=logger, bins=bins)
