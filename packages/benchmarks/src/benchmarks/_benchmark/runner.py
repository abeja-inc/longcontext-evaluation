import logging
from pathlib import Path
from typing import Any

from llm_inference import BaseGenerator
from tqdm import tqdm

from .config import RunOptions
from .core import append_jsonl
from .interfaces import PredictJob
from .result import push_results_to_wandb, save_results


class BenchmarkRunner:
    def __init__(
        self,
        *,
        generator: BaseGenerator,
        generate_kwargs: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.generator = generator
        self.generate_kwargs = generate_kwargs
        self.logger = logger

    def run(self, *, jobs: list[PredictJob], model_name: str, opts: RunOptions) -> None:
        if not jobs:
            raise ValueError("jobs is empty.")

        pred_dir = Path(opts.prediction_dir).expanduser().resolve()
        pred_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_under_prediction_dir(pred_path: Path) -> Path:
            p = pred_path.expanduser().resolve()
            try:
                p.relative_to(pred_dir)
            except Exception as e:
                raise ValueError(
                    f"job.pred_path must be under prediction_dir.\n"
                    f"  prediction_dir: {pred_dir}\n"
                    f"  pred_path:       {p}"
                ) from e
            return p

        # ---- PREDICT phase ----
        for job in jobs:
            job_pred_path = _ensure_under_prediction_dir(job.pred_path)
            job_pred_path.parent.mkdir(parents=True, exist_ok=True)
            job_pred_path.touch(exist_ok=True)

            processed = job.load_processed_ids()
            self.logger.info(
                "[PREDICT] job=%s pred=%s processed=%d",
                job.name,
                job_pred_path,
                len(processed),
            )

            batches = list(job.iter_batches(opts.batch_size))
            for batch in tqdm(batches, desc=f"Predict {job.name}"):
                records = job.run_batch(
                    generator=self.generator,
                    generate_kwargs=self.generate_kwargs,
                    batch=batch,
                )
                for r in records:
                    append_jsonl(job_pred_path, r)

        # ---- EVAL phase (optional) ----
        summary_json: dict[str, Any] | None = None
        if opts.evaluator is not None:
            eval_output_path = (
                opts.eval_output_path.expanduser().resolve()
                if opts.eval_output_path is not None
                else (pred_dir / "summary.json")
            )
            eval_output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                "[EVAL] evaluator=%s prediction_dir=%s out=%s",
                opts.evaluator.name,
                pred_dir,
                eval_output_path,
            )

            summary_json = opts.evaluator.run(
                prediction_dir=pred_dir,
                output_path=eval_output_path,
            )

        # ---- RESULTS phase (optional) ----
        results = None
        if opts.results_builder is not None:
            self.logger.info(
                "[RESULTS] builder=%s prediction_dir=%s",
                opts.results_builder.name,
                pred_dir,
            )
            results = opts.results_builder.build_results(
                model_name=model_name,
                prediction_dir=pred_dir,
                summary_json=summary_json,
            )

            # Save locally (optional)
            if opts.results_output_dir is not None:
                results_out = opts.results_output_dir.expanduser().resolve()
                results_out.mkdir(parents=True, exist_ok=True)

                manifest_path = save_results(
                    results=results,
                    output_dirpath=results_out,
                    format=opts.results_format,
                )
                self.logger.info("[RESULTS] saved manifest: %s", manifest_path)

        # ---- W&B phase (optional) ----
        if opts.wandb_run is not None:
            if results is None:
                if opts.results_builder is None:
                    raise ValueError("wandb_run is set but results_builder is None.")
                results = opts.results_builder.build_results(
                    model_name=model_name,
                    prediction_dir=pred_dir,
                    summary_json=summary_json,
                )
            push_results_to_wandb(results=results, wandb_run=opts.wandb_run)
