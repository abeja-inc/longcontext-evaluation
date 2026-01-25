from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from llm_inference.base import BaseGenerator

from .result import Results


@dataclass(frozen=True)
class Batch:
    samples: list[dict[str, Any]]


class PredictJob(Protocol):
    """
    Job は「どう読んで」「どう入力を作って」「どう出力レコードを作るか」を握る。
    Runner は job を回すだけ。
    """

    name: str
    pred_path: Path

    def load_processed_ids(self) -> set[str]: ...
    def iter_batches(self, batch_size: int) -> Iterable[Batch]: ...
    def run_batch(
        self,
        *,
        generator: BaseGenerator,
        generate_kwargs: dict[str, Any],
        batch: Batch,
    ) -> list[dict[str, Any]]: ...


class Evaluator(Protocol):
    """
    予測ファイル群（pred_dir）から summary.json などを作る。
    MRCR/RULER の EvaluationPipeline を薄く包む用途。
    """

    name: str

    def run(
        self, *, prediction_dir: Path, output_path: Path
    ) -> dict[str, Any] | None: ...


class ResultsBuilder(Protocol):
    """
    ベンチ側が提供する「Results への変換器」。
    - LongBench: prediction_dir の jsonl を読んで採点・集計して Results を作る
    - MRCR/RULER: evaluator が作った summary_json を flatten して Results を作る、など
    """

    name: str

    def build_results(
        self,
        *,
        model_name: str,
        prediction_dir: Path,
        summary_json: dict[str, Any] | None,
    ) -> Results: ...
