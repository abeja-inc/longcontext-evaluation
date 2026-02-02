import importlib
import inspect
from logging import Logger
from typing import Type

from .base import BaseGenerator


def get_generator(type: str, *, logger: Logger, **kwargs) -> BaseGenerator:
    logger.info(f"Initializing generator of type '{type}'")

    module_path = f"{__package__}.{type}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(f"Generator module '{type}.py' not found") from e

    generator_classes: list[Type[BaseGenerator]] = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseGenerator)
            and obj is not BaseGenerator
            and obj.__module__ == module.__name__  # 他モジュール混入防止
        ):
            generator_classes.append(obj)

    if not generator_classes:
        raise ValueError(f"No BaseGenerator subclass found in '{type}.py'")

    if len(generator_classes) > 1:
        raise ValueError(
            f"Multiple BaseGenerator subclasses found in '{type}.py': "
            f"{[cls.__name__ for cls in generator_classes]}"
        )

    cls = generator_classes[0]
    return cls(logger=logger, **kwargs)
