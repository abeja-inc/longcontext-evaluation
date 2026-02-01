from logging import Logger
from typing import Literal

from ._core.evaluate import Bin
from ._core.runner import BaseBenchmarkRunner
from .longbench_v2 import LongBenchV2Runner


RUNNER = Literal["longbench_v2"]

RUNNER_REGISTRY = {
    "longbench_v2": LongBenchV2Runner,
}


def get_runner(
    name: RUNNER, logger: Logger, bins: list[Bin] | None = None
) -> BaseBenchmarkRunner:
    return RUNNER_REGISTRY[name](logger=logger, bins=bins)
