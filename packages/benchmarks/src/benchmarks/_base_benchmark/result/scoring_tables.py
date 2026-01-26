from collections import defaultdict
from typing import Any, Iterable

from ...data_model import Score

CONTEXT_LENGTH_BINS: list[tuple[int, str]] = [
    (4096, "~4k"),
    (8192, "~8k"),
    (16384, "~16k"),
    (32768, "~32k"),
    (65536, "~64k"),
    (131072, "~128k"),
]


def context_length_bin(context_length: int | None) -> str:
    if not isinstance(context_length, int) or context_length <= 0:
        return "unknown"
    for limit, label in CONTEXT_LENGTH_BINS:
        if context_length <= limit:
            return label
    return "over128k"


def score_rows(scores: Iterable[Score], *, model_name: str) -> list[dict[str, Any]]:
    return [
        {"Model": model_name, **score.model_dump()}
        for score in scores
    ]


def _mean_score(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def leaderboard_rows(scores: Iterable[Score], *, model_name: str) -> list[dict[str, Any]]:
    scores = list(scores)
    rows: list[dict[str, Any]] = []

    by_task: dict[tuple[str, str, str], list[Score]] = defaultdict(list)
    by_subset: dict[tuple[str, str, str, str], list[Score]] = defaultdict(list)
    by_context: dict[tuple[str, str, str, str, str], list[Score]] = defaultdict(list)

    for score in scores:
        by_task[(score.benchmark, score.language, score.task)].append(score)
        by_subset[(score.benchmark, score.language, score.task, score.subset)].append(
            score
        )
        by_context[
            (
                score.benchmark,
                score.language,
                score.task,
                score.subset,
                context_length_bin(score.context_length),
            )
        ].append(score)

    for (benchmark, language, task), items in sorted(by_task.items()):
        values = [s.score for s in items]
        rows.append(
            {
                "Model": model_name,
                "Benchmark": benchmark,
                "Language": language,
                "Task": task,
                "Subset": "ALL",
                "Context Length Bin": "ALL",
                "mean_score": _mean_score(values),
                "n": len(values),
                "group": "task_mean",
            }
        )

    for (benchmark, language, task, subset), items in sorted(by_subset.items()):
        values = [s.score for s in items]
        rows.append(
            {
                "Model": model_name,
                "Benchmark": benchmark,
                "Language": language,
                "Task": task,
                "Subset": subset,
                "Context Length Bin": "ALL",
                "mean_score": _mean_score(values),
                "n": len(values),
                "group": "subset_mean",
            }
        )

    for (benchmark, language, task, subset, ctx_bin), items in sorted(
        by_context.items()
    ):
        values = [s.score for s in items]
        rows.append(
            {
                "Model": model_name,
                "Benchmark": benchmark,
                "Language": language,
                "Task": task,
                "Subset": subset,
                "Context Length Bin": ctx_bin,
                "mean_score": _mean_score(values),
                "n": len(values),
                "group": "context_length_mean",
            }
        )

    return rows


def score_by_context_length_rows(
    scores: Iterable[Score], *, model_name: str
) -> list[dict[str, Any]]:
    scores = list(scores)
    by_context: dict[tuple[str, str, str, str, str], list[Score]] = defaultdict(list)
    for score in scores:
        by_context[
            (
                score.benchmark,
                score.language,
                score.task,
                score.subset,
                context_length_bin(score.context_length),
            )
        ].append(score)

    rows: list[dict[str, Any]] = []
    for (benchmark, language, task, subset, ctx_bin), items in sorted(
        by_context.items()
    ):
        values = [s.score for s in items]
        rows.append(
            {
                "Model": model_name,
                "Benchmark": benchmark,
                "Language": language,
                "Task": task,
                "Subset": subset,
                "Context Length Bin": ctx_bin,
                "mean_score": _mean_score(values),
                "n": len(values),
            }
        )

    return rows


def summary_from_scores(scores: Iterable[Score]) -> dict[str, Any]:
    scores = list(scores)
    summary: dict[str, Any] = {}
    values = [s.score for s in scores]
    summary["mean/overall_score"] = _mean_score(values)

    by_task: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        by_task[str(score.task)].append(score.score)
    for task, task_scores in by_task.items():
        summary[f"mean/{task}"] = _mean_score(task_scores)
    return summary
