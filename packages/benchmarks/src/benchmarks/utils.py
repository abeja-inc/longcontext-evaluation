import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence


def get_custom_logger(name: str = "benchmarks", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_child_logger(parent: logging.Logger, name: str) -> logging.Logger:
    child = parent.getChild(name)
    if not child.handlers:
        for handler in parent.handlers:
            child.addHandler(handler)
    child.setLevel(parent.level)
    return child


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _matches_patterns(name: str, patterns: Sequence[str], *, allow_stem: bool) -> bool:
    if not patterns:
        return True
    candidates = [name]
    if allow_stem:
        candidates.append(Path(name).stem)
    return any(fnmatch(candidate, pattern) for pattern in patterns for candidate in candidates)


def filter_names(
    names: Iterable[str],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    *,
    allow_stem: bool = False,
) -> list[str]:
    include = include or []
    exclude = exclude or []
    filtered: list[str] = []
    for name in names:
        if include and not _matches_patterns(name, include, allow_stem=allow_stem):
            continue
        if exclude and _matches_patterns(name, exclude, allow_stem=allow_stem):
            continue
        filtered.append(name)
    return filtered
