from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .interfaces import Evaluator, ResultsBuilder


@dataclass
class RunOptions:
    batch_size: int
    prediction_dir: Path

    # optional eval
    evaluator: Evaluator | None = None
    eval_output_path: Path | None = None  # e.g. <prediction_dir>/summary.json

    # optional results
    results_builder: ResultsBuilder | None = None
    results_output_dir: Path | None = None  # e.g. <output_dir>/results
    results_format: Literal["jsonl", "csv"] = "jsonl"

    # optional wandb
    wandb_run: Any | None = None
