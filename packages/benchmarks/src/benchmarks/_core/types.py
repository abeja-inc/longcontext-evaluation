from logging import Logger
from typing import Protocol

from llm_inference.base import BaseGenerator

from ..config import BenchmarkConfig
from .evaluate import Bin


class Runner(Protocol):
    def run(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, object],
        config: BenchmarkConfig,
        batchsize: int,
        log_wandb: bool,
    ) -> None: ...


class RunnerConstructor(Protocol):
    def __call__(self, *, logger: Logger, bins: list[Bin] | None = None) -> Runner: ...
