from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from llm_inference.base import BaseGenerator
from transformers import AutoTokenizer

from .._base_benchmark.config import RunOptions
from .._base_benchmark.result import push_results_to_wandb
from .._base_benchmark.runner import BenchmarkRunner
from ..utils import parse_csv_list
from .prediction.predict import build_jobs_for_dataset_dir, load_prompt_templates
from .evaluation.results import LongBenchResultsBuilder


def run_longbench_v2(
    *,
    cfg: dict[str, Any],
    generator: BaseGenerator,
    generate_kwargs: dict[str, Any],
    model_name: str,
    model_root: Path,
    dataset_dir: Path,
    prediction_dir: Path,
    batchsize: int,
    max_model_len: int,
    max_new_tokens: int,
    logger: logging.Logger,
    wandb_run_factory: Callable[[dict[str, Any], str], Any],
) -> None:
    if not cfg.get("enabled", True):
        return

    prompt_templates = load_prompt_templates(Path(cfg["tasks_path"]).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(str(model_root / model_name))

    jobs = build_jobs_for_dataset_dir(
        dataset_dir=dataset_dir,
        prediction_dir=prediction_dir,
        prompt_templates=prompt_templates,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        rag_topn=cfg.get("rag_topn", 0),
        cot=cfg.get("cot", False),
        no_context=cfg.get("no_context", False),
        include_datasets=parse_csv_list(cfg.get("only_datasets")),
        exclude_datasets=parse_csv_list(cfg.get("exclude_datasets")),
    )

    if cfg.get("run_pred", True):
        runner = BenchmarkRunner(
            generator=generator, generate_kwargs=generate_kwargs, logger=logger
        )
        runner.run(
            jobs=jobs,
            model_name=model_name,
            opts=RunOptions(batch_size=batchsize, prediction_dir=prediction_dir),
        )

    if not cfg.get("run_eval", True):
        return

    results_builder = LongBenchResultsBuilder(
        compensate_missing=cfg.get("compensate_missing", False)
    )
    results = results_builder.build_results(
        model_name=model_name, prediction_dir=prediction_dir, summary_json=None
    )

    run = wandb_run_factory(cfg.get("wandb", {}), "LongBench-v2")
    if run and push_results_to_wandb is not None:
        push_results_to_wandb(results=results, wandb_run=run)
    if run is not None:
        run.finish()
