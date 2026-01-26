import argparse
import logging
from pathlib import Path
from typing import Any, Callable

import wandb
import yaml
from openai import OpenAI
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator
from llm_inference.vllm_openai_api_compatible import VLLMOpenAICompatibleGenerator
from llm_inference.openai_api import OpenAIGenerator
from vllm import SamplingParams

from benchmarks.longbench_v2.runner import run_longbench_v2
from benchmarks.mrcr.runner import run_mrcr
from benchmarks.ruler.runner import run_ruler
from benchmarks.utils import filter_names, get_custom_logger, parse_csv_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("./run_config.yml"),
        help="Path to benchmark run config",
    )
    p.add_argument(
        "--only-benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names to include",
    )
    p.add_argument(
        "--exclude-benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names to exclude",
    )
    return p.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_generator(
    *, config: dict[str, Any], model_root: Path, model_name: str, logger: logging.Logger
) -> tuple[Any, dict[str, Any], int, int]:
    gen_cfg = config.get("generator", {})
    gen_type = gen_cfg.get("type", "vllm_offline")

    if gen_type == "vllm_offline":
        vllm_config_raw = gen_cfg.get("vllm_config", {})
        if isinstance(vllm_config_raw, dict):
            vllm_config = vllm_config_raw
        else:
            vllm_config_path = expand_path(vllm_config_raw or "./vllm_offline_config.yml")
            with vllm_config_path.open("r", encoding="utf-8") as f:
                vllm_config = yaml.safe_load(f)

        serve_cfg: dict[str, Any] = vllm_config.get("serve", {})
        generation_cfg: dict[str, Any] = vllm_config.get("generation", {})

        extra_args = serve_cfg.get("extra_args", {})
        max_model_len = extra_args.get(
            "max_model_len", serve_cfg.get("max_model_len", 4096)
        )
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
            model_name=str(model_root / model_name),
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

    if gen_type == "openai":
        api_key = gen_cfg.get("api_key")
        base_url = gen_cfg.get("base_url")
        client = OpenAI(api_key=api_key, base_url=base_url)
        model_name = gen_cfg.get("model_name")
        if not model_name:
            raise ValueError("generator.model_name is required for openai generator.")
        max_model_len = int(gen_cfg.get("max_context_length", 8192))
        max_new_tokens = int(gen_cfg.get("max_output_tokens", 512))
        generator = OpenAIGenerator(
            client=client,
            model_name=model_name,
            max_context_length=max_model_len,
            max_output_tokens=max_new_tokens,
            logger=logger,
        )
        generate_kwargs = {
            "buffer_tokens": int(gen_cfg.get("buffer_tokens", 10)),
            "chat_kwargs": gen_cfg.get("chat_kwargs", {}),
            "completion_kwargs": gen_cfg.get("completion_kwargs", {}),
        }
        return generator, generate_kwargs, max_model_len, max_new_tokens

    if gen_type == "vllm_openai_compatible":
        api_key = gen_cfg.get("api_key")
        base_url = gen_cfg.get("base_url")
        client = OpenAI(api_key=api_key, base_url=base_url)
        model_name = gen_cfg.get("model_name")
        model_path = gen_cfg.get("model_path")
        if not model_name or not model_path:
            raise ValueError(
                "generator.model_name and generator.model_path are required for vllm_openai_compatible."
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(expand_path(model_path)))
        max_model_len = int(gen_cfg.get("max_context_length", 8192))
        max_new_tokens = int(gen_cfg.get("max_output_tokens", 512))
        generator = VLLMOpenAICompatibleGenerator(
            client=client,
            tokenizer=tokenizer,
            model_name=model_name,
            max_context_length=max_model_len,
            max_output_tokens=max_new_tokens,
            logger=logger,
        )
        generate_kwargs = {
            "buffer_tokens": int(gen_cfg.get("buffer_tokens", 10)),
            "chat_kwargs": gen_cfg.get("chat_kwargs", {}),
            "completion_kwargs": gen_cfg.get("completion_kwargs", {}),
        }
        return generator, generate_kwargs, max_model_len, max_new_tokens

    raise ValueError(f"Unsupported generator.type: {gen_type}")


def build_wandb_factory(model_name: str) -> Callable[[dict[str, Any], str], Any]:
    def _init_wandb(wandb_cfg: dict[str, Any], group: str):
        project = wandb_cfg.get("project")
        if not project:
            return None
        return wandb.init(
            project=project,
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name") or model_name,
            group=wandb_cfg.get("group") or group,
            config=wandb_cfg.get("config", {}),
        )

    return _init_wandb


def main() -> None:
    args = parse_args()
    config_path = expand_path(args.config)
    logger = get_custom_logger()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_config_path = config.get("base_config")
    if base_config_path:
        base_config_path = Path(base_config_path)
        if not base_config_path.is_absolute():
            base_config_path = config_path.parent / base_config_path
        base_config_path = expand_path(base_config_path)
        with base_config_path.open("r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        config = merge_dicts(base_config, config)

    model_cfg = config.get("model", {})
    model_root = expand_path(model_cfg.get("root", "./models"))
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("model.name is required in config.")

    output_dir = expand_path(config.get("output_dir", "./outputs"))
    dataset_dir = expand_path(config.get("dataset_dir", "./datasets"))
    batchsize = int(config.get("batchsize", 1))
    generator, generate_kwargs, max_model_len, max_new_tokens = build_generator(
        config=config,
        model_root=model_root,
        model_name=model_name,
        logger=logger,
    )

    wandb_run_factory = build_wandb_factory(model_name)

    only_benchmarks = parse_csv_list(args.only_benchmarks)
    exclude_benchmarks = parse_csv_list(args.exclude_benchmarks)

    benchmarks_cfg = config.get("benchmarks", {})
    benchmark_names = filter_names(
        benchmarks_cfg.keys(), include=only_benchmarks, exclude=exclude_benchmarks
    )

    prediction_dir = output_dir / model_name

    for benchmark_name in benchmark_names:
        bench_cfg = dict(benchmarks_cfg.get(benchmark_name, {}))
        bench_cfg["_config_dir"] = config_path.parent
        if benchmark_name == "longbench_v2":
            run_longbench_v2(
                cfg=bench_cfg,
                generator=generator,
                generate_kwargs=generate_kwargs,
                model_name=model_name,
                model_root=model_root,
                dataset_dir=dataset_dir,
                prediction_dir=prediction_dir,
                batchsize=batchsize,
                max_model_len=max_model_len,
                max_new_tokens=max_new_tokens,
                logger=logger,
                wandb_run_factory=wandb_run_factory,
            )
        elif benchmark_name == "mrcr":
            run_mrcr(
                cfg=bench_cfg,
                generator=generator,
                generate_kwargs=generate_kwargs,
                model_name=model_name,
                dataset_dir=dataset_dir,
                prediction_dir=prediction_dir,
                batchsize=batchsize,
                logger=logger,
                wandb_run_factory=wandb_run_factory,
            )
        elif benchmark_name == "ruler":
            run_ruler(
                cfg=bench_cfg,
                generator=generator,
                generate_kwargs=generate_kwargs,
                model_name=model_name,
                model_root=model_root,
                dataset_dir=dataset_dir,
                prediction_dir=prediction_dir,
                batchsize=batchsize,
                logger=logger,
                wandb_run_factory=wandb_run_factory,
            )
        else:
            logger.warning("Unknown benchmark: %s", benchmark_name)


if __name__ == "__main__":
    main()
