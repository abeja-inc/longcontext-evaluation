from __future__ import annotations

from pathlib import Path
from typing import Any

from ..._base_benchmark.core import read_jsonl
from ..._base_benchmark.interfaces import ResultsBuilder
from ..._base_benchmark.result import Results, Table
from .scoring import LongBenchScorer


class LongBenchResultsBuilder(ResultsBuilder):
    name = "longbench_v2"

    def __init__(self, *, compensate_missing: bool = False) -> None:
        self._compensate_missing = compensate_missing
        self._scorer = LongBenchScorer(compensate_missing=compensate_missing)

    def build_results(
        self,
        *,
        model_name: str,
        prediction_dir: Path,
        summary_json: dict[str, Any] | None,
    ) -> Results:
        files = sorted(prediction_dir.rglob("*.jsonl"))

        metrics_rows: list[dict[str, Any]] = []
        prediction_tables: list[Table] = []

        for filepath in files:
            records = read_jsonl(filepath)
            metrics, rows = self._scorer.score_records(records)
            prompt_type = filepath.parent.name
            subset_name = filepath.stem

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Prompt Type": prompt_type,
                    "Subset": subset_name,
                    **metrics,
                }
            )

            if rows:
                subset_rows = [
                    {
                        "Model": model_name,
                        "Prompt Type": prompt_type,
                        "Subset": subset_name,
                        **r,
                    }
                    for r in rows
                ]
                prediction_tables.append(
                    Table(
                        name=f"predictions/{prompt_type}/{subset_name}",
                        rows=subset_rows,
                    )
                )

        mean_metrics = _mean_metrics(model_name=model_name, rows=metrics_rows)
        summary = _summary_from_mean(mean_metrics)

        tables: list[Table] = [
            Table(name="metrics", rows=metrics_rows),
            Table(name="metrics_mean", rows=[mean_metrics] if mean_metrics else []),
            *prediction_tables,
        ]

        return Results(
            tables=tables,
            summary=summary,
            config={
                "model_name": model_name,
                "prediction_dir": str(prediction_dir),
                "compensate_missing": self._compensate_missing,
            },
        )


def _mean_metrics(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    percent_keys = [
        "overall_acc",
        "easy_acc",
        "hard_acc",
        "short_acc",
        "medium_acc",
        "long_acc",
    ]

    mean_metrics: dict[str, Any] = {
        "Model": model_name,
        "Prompt Type": "ALL",
        "Subset": "ALL",
    }

    mean_metrics["overall_n"] = sum(r.get("overall_n", 0) for r in rows)
    mean_metrics["easy_n"] = sum(r.get("easy_n", 0) for r in rows)
    mean_metrics["hard_n"] = sum(r.get("hard_n", 0) for r in rows)
    mean_metrics["short_n"] = sum(r.get("short_n", 0) for r in rows)
    mean_metrics["medium_n"] = sum(r.get("medium_n", 0) for r in rows)
    mean_metrics["long_n"] = sum(r.get("long_n", 0) for r in rows)

    for key in percent_keys:
        vals = [r.get(key, 0.0) for r in rows if r.get("overall_n", 0) > 0]
        mean_metrics[key] = round(sum(vals) / len(vals), 2) if vals else 0.0

    return mean_metrics


def _summary_from_mean(mean_metrics: dict[str, Any]) -> dict[str, Any]:
    if not mean_metrics:
        return {}

    summary: dict[str, Any] = {
        "model/mean_overall_acc": mean_metrics.get("overall_acc", 0.0),
        "mean/overall_n": mean_metrics.get("overall_n", 0),
    }

    for key in [
        "overall_acc",
        "easy_acc",
        "hard_acc",
        "short_acc",
        "medium_acc",
        "long_acc",
    ]:
        summary[f"mean/{key}"] = mean_metrics.get(key, 0.0)

    return summary
