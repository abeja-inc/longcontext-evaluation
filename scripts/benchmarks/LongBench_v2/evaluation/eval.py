import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator
from transformers import AutoTokenizer
from vllm import SamplingParams

from benchmarks._base_benchmark.config import RunOptions
from benchmarks._base_benchmark.runner import BenchmarkRunner
from benchmarks.longbench_v2.predict import build_jobs_for_dataset_dir, load_prompt_templates
from benchmarks.utils import get_custom_logger, parse_csv_list


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-root", type=Path, required=True, help="Path to models root directory"
    )
    p.add_argument("--model-name", type=str, required=True, help="model name")
    p.add_argument(
        "--vllm-config",
        type=Path,
        default=Path("./vllm_offline_config.yml"),
        help="Path to vllm config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/longcontext-llm-benchmark-eval/experimentation/outputs"),
        help="Path to output directory",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("~/datasets"),
        help="Path to dataset directory",
    )
    p.add_argument(
        "--tasks", type=Path, default=Path("./tasks.yml"), help="Path to tasks file"
    )
    p.add_argument(
        "--rag-topn",
        type=int,
        default=0,
        help="Use top-N retrieved chunks if dataset has them",
    )
    p.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    p.add_argument("--no-context", action="store_true", help="Do not use context")
    p.add_argument("--batchsize", type=int, default=1, help="Batch size for inference")
    p.add_argument(
        "--only-datasets",
        type=str,
        default=None,
        help="Comma-separated dataset name patterns to include",
    )
    p.add_argument(
        "--exclude-datasets",
        type=str,
        default=None,
        help="Comma-separated dataset name patterns to exclude",
    )
    return p.parse_args()


def build_generator(
    *, model_path: Path, vllm_config_path: Path, logger: logging.Logger
) -> tuple[VLLMOfflineGenerator, dict[str, Any], int, int]:
    with vllm_config_path.open("r", encoding="utf-8") as f:
        vllm_config = yaml.safe_load(f)

    serve_cfg: dict[str, Any] = vllm_config.get("serve", {})
    generation_cfg: dict[str, Any] = vllm_config.get("generation", {})

    extra_args = serve_cfg.get("extra_args", {})
    max_model_len = extra_args.get("max_model_len", serve_cfg.get("max_model_len", 4096))
    max_new_tokens = generation_cfg.get(
        "max_new_tokens",
        generation_cfg.get("max_output_tokens", generation_cfg.get("max_tokens", 512)),
    )

    reasoning_parser = serve_cfg.get("reasoning_parser")
    generator_kwargs = {
        k: v
        for k, v in serve_cfg.items()
        if k
        not in {
            "extra_args",
            "model_name_or_path",
            "max_model_len",
            "reasoning_parser",
        }
    }

    generator = VLLMOfflineGenerator(
        model_name=str(model_path),
        max_context_length=max_model_len,
        max_output_tokens=max_new_tokens,
        logger=logger,
        reasoning_parser=reasoning_parser,
        **generator_kwargs,
        **extra_args,
    )

    sampling_params = SamplingParams(
        **{
            k: v
            for k, v in generation_cfg.items()
            if k
            not in {
                "max_new_tokens",
                "max_output_tokens",
                "max_tokens",
                "chat_template_kwargs",
                "buffer_tokens",
            }
        }
    )

    generate_kwargs = {
        "sampling_params": sampling_params,
        "buffer_tokens": generation_cfg.get("buffer_tokens", 10),
        "chat_template_kwargs": generation_cfg.get("chat_template_kwargs", {}),
    }

    return generator, generate_kwargs, max_model_len, max_new_tokens


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    model_path: Path = expand_path(args.model_root) / args.model_name
    dataset_dirpath: Path = expand_path(args.dataset_dir)
    output_dirpath: Path = expand_path(args.output_dir) / args.model_name

    generator, generate_kwargs, max_model_len, max_new_tokens = build_generator(
        model_path=model_path, vllm_config_path=args.vllm_config, logger=logger
    )

    prompt_templates = load_prompt_templates(args.tasks)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    jobs = build_jobs_for_dataset_dir(
        dataset_dir=dataset_dirpath,
        prediction_dir=output_dirpath,
        prompt_templates=prompt_templates,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        rag_topn=args.rag_topn,
        cot=args.cot,
        no_context=args.no_context,
        include_datasets=parse_csv_list(args.only_datasets),
        exclude_datasets=parse_csv_list(args.exclude_datasets),
    )

    if not jobs:
        logger.warning("No datasets found under %s", dataset_dirpath)
        return

    runner = BenchmarkRunner(
        generator=generator, generate_kwargs=generate_kwargs, logger=logger
    )
    runner.run(
        jobs=jobs,
        model_name=args.model_name,
        opts=RunOptions(batch_size=args.batchsize, prediction_dir=output_dirpath),
    )


if __name__ == "__main__":
    args = parse_args()
    logger = get_custom_logger()
    main(args=args, logger=logger)
