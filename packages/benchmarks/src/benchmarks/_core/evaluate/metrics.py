from abc import ABC
from logging import Logger
from typing import Any, Generic, TypeVar

from ...config import SubtaskConfig
from ..predict.data import OutputType
from ..settings import SettingsType


class BaseMetrics(ABC, Generic[SettingsType, OutputType]):
    _metric_registry: dict[str, str] = {}

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def eval(
        self,
        output: OutputType,
        config: SubtaskConfig,
        settings: SettingsType,
        **kwargs: Any,
    ) -> float:
        method_name = self._metric_registry[config.metric]
        method = getattr(self, method_name)
        return method(output=output, config=config, settings=settings, **kwargs)


MetricsType = TypeVar("MetricsType", bound=BaseMetrics)
