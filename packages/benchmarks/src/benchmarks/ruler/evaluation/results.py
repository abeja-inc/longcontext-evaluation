from __future__ import annotations

from pathlib import Path
from typing import Any

from ..._base_benchmark.core import read_jsonl
from ..._base_benchmark.interfaces import ResultsBuilder
from ..._base_benchmark.result import Results, Table
from ..._base_benchmark.result.scoring_tables import (
    leaderboard_rows,
    score_by_context_length_rows,
    summary_from_scores,
)
from ...data_model import Score


class RulerResultsBuilder(ResultsBuilder):
    name = "ruler"

    def build_results(
        self,
        *,
        model_name: str,
        prediction_dir: Path,
        summary_json: dict[str, Any] | None,
    ) -> Results:
        summary_json = summary_json or {}
        scores = _flatten_scores(summary=summary_json)
        leaderboard = leaderboard_rows(scores, model_name=model_name)
        by_context = score_by_context_length_rows(scores, model_name=model_name)
        output_rows = _build_output_rows(
            model_name=model_name, prediction_dir=prediction_dir, scores=scores
        )
        summary = summary_from_scores(scores)

        tables = [
            Table(name="table/ruler_output_table", rows=output_rows),
            Table(name="metrics/ruler_leaderboard", rows=leaderboard),
            Table(name="metrics/ruler_score_by_context_length", rows=by_context),
        ]
        return Results(
            tables=tables,
            summary=summary,
            config={
                "model_name": model_name,
                "prediction_dir": str(prediction_dir),
            },
        )


def _flatten_scores(summary: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    for task_obj in summary.get("results", []):
        for subset in task_obj.get("subsets", []):
            for score in subset.get("score", []):
                scores.append(Score.model_validate(score))
    return scores


def _build_output_rows(
    *,
    model_name: str,
    prediction_dir: Path,
    scores: list[Score],
) -> list[dict[str, Any]]:
    score_map = {
        (score.task, score.subset, score.index): score for score in scores
    }
    rows: list[dict[str, Any]] = []
    for filepath in sorted(prediction_dir.rglob("*.jsonl")):
        task = filepath.parent.name
        subset = filepath.stem
        records = read_jsonl(filepath)
        for index, record in enumerate(records):
            score = score_map.get((task, subset, index))
            rows.append(
                {
                    "Model": model_name,
                    "Task": task,
                    "Subset": subset,
                    "index": index,
                    "score": score.score if score else None,
                    "context_length": score.context_length if score else None,
                    **record,
                }
            )
    return rows
