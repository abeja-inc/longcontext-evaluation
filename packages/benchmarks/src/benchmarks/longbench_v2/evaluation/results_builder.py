from pathlib import Path
from typing import Any

from ..._core.interfaces import ResultsBuilder
from ..._core.utils import read_jsonl
from ..._scoring import Results, Score, Table
from ..._scoring.scoring_tables import (
    leaderboard_rows,
    score_by_context_length_rows,
    summary_from_scores,
)
from .scorer import build_scores


class LongBenchResultsBuilder(ResultsBuilder):
    name = "longbench_v2"

    def __init__(self, *, compensate_missing: bool = False) -> None:
        self._compensate_missing = compensate_missing

    def build_results(
        self,
        *,
        model_name: str,
        prediction_dir: Path,
        summary_json: dict[str, Any] | None,
    ) -> Results:
        files = sorted(prediction_dir.rglob("*.jsonl"))

        scores = _flatten_scores(summary_json) if summary_json else []
        output_rows: list[dict[str, Any]] = []

        for filepath in files:
            records = read_jsonl(filepath)
            prompt_type = filepath.parent.name
            subset_name = filepath.stem
            file_scores, rows = build_scores(
                records,
                prompt_type=prompt_type,
                subset_name=subset_name,
                compensate_missing=self._compensate_missing,
            )
            if not summary_json:
                scores.extend(file_scores)
            for record, row in zip(records, rows, strict=False):
                context_length = row.get("token_count")
                output_rows.append(
                    {
                        "Model": model_name,
                        "Prompt Type": prompt_type,
                        "Subset": subset_name,
                        "score": row.get("acc", 0.0),
                        "context_length": context_length,
                        **row,
                        **record,
                    }
                )

        leaderboard = leaderboard_rows(scores, model_name=model_name)
        by_context = score_by_context_length_rows(scores, model_name=model_name)
        summary = summary_from_scores(scores)

        tables: list[Table] = [
            Table(name="table/longbench_v2_output_table", rows=output_rows),
            Table(name="metrics/longbench_v2_leaderboard", rows=leaderboard),
            Table(
                name="metrics/longbench_v2_score_by_context_length",
                rows=by_context,
            ),
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


def _flatten_scores(summary: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    for task_obj in summary.get("results", []):
        for subset in task_obj.get("subsets", []):
            for score in subset.get("score", []):
                scores.append(Score.model_validate(score))
    return scores
