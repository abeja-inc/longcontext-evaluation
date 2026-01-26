from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..._base_benchmark.interfaces import ResultsBuilder
from ..._base_benchmark.result import Results, Table


class MRCRResultsBuilder(ResultsBuilder):
    name = "mrcr"

    def build_results(
        self,
        *,
        model_name: str,
        prediction_dir: Path,
        summary_json: dict[str, Any] | None,
    ) -> Results:
        summary_json = summary_json or {}
        flat_rows = _flatten_rows(model_name=model_name, summary=summary_json)
        task_means = _mean_by_task(model_name=model_name, rows=flat_rows)
        summary = _build_summary(flat_rows=flat_rows, mean_rows=task_means)

        tables = [
            Table(name="scores", rows=flat_rows),
            Table(name="task_means", rows=task_means),
        ]
        return Results(
            tables=tables,
            summary=summary,
            config={
                "model_name": model_name,
                "prediction_dir": str(prediction_dir),
            },
        )


def _flatten_rows(model_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for task_obj in summary.get("results", []):
        task = task_obj.get("task")
        for subset in task_obj.get("subsets", []):
            subset_name = subset.get("subset_name")
            for s in subset.get("score", []):
                flat.append(
                    {
                        "Model": model_name,
                        "Task": task,
                        "Subset": subset_name,
                        "context_length": s.get("context_length"),
                        "score": s.get("score"),
                    }
                )
    return flat


def _mean_score(rows: list[dict[str, Any]]) -> float:
    vals = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _mean_by_task(
    *, model_name: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("Task"))].append(row)

    mean_rows: list[dict[str, Any]] = []
    for task, task_rows in by_task.items():
        mean_rows.append(
            {"Model": model_name, "Task": task, "mean_score": _mean_score(task_rows)}
        )
    return mean_rows


def _build_summary(
    *, flat_rows: list[dict[str, Any]], mean_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["mean/overall_score"] = _mean_score(flat_rows)
    for row in mean_rows:
        task = row.get("Task")
        summary[f"mean/{task}"] = row.get("mean_score", 0.0)
    return summary
