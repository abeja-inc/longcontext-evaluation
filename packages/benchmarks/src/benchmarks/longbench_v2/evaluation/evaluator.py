import json
from pathlib import Path
from typing import Any

from ..._core.interfaces import Evaluator
from ..._core.utils import read_jsonl
from .scorer import LongBenchScorer


class LongBenchEvaluator(Evaluator):
    name = "longbench_v2"

    def __init__(self, *, compensate_missing: bool = False) -> None:
        self._scorer = LongBenchScorer(compensate_missing=compensate_missing)

    def run(self, *, prediction_dir: Path, output_path: Path) -> dict[str, Any] | None:
        summary: dict[str, Any] = {"results": []}
        files = sorted(prediction_dir.rglob("*.jsonl"))
        for filepath in files:
            records = read_jsonl(filepath)
            metrics, rows = self._scorer.score_records(records)
            summary["results"].append(
                {
                    "task": filepath.parent.name,
                    "subset": filepath.stem,
                    "metrics": metrics,
                    "rows": rows,
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
