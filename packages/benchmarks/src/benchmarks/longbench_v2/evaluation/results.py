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

        scores: list[Score] = []
        output_rows: list[dict[str, Any]] = []

        for filepath in files:
            records = read_jsonl(filepath)
            metrics, rows = self._scorer.score_records(records)
            prompt_type = filepath.parent.name
            subset_name = filepath.stem
            for index, (record, row) in enumerate(zip(records, rows, strict=False)):
                context_length = row.get("token_count")
                scores.append(
                    Score(
                        benchmark=self.name,
                        language=record.get("language", "unknown"),
                        task=prompt_type,
                        subset=subset_name,
                        index=row.get("sample_id", index),
                        score=row.get("acc", 0.0),
                        context_length=context_length if context_length is not None else -1,
                        tags=record.get("tags"),
                    )
                )
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
