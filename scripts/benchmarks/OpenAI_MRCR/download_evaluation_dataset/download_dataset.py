import argparse
import json
import logging
from pathlib import Path

import yaml
from benchmarks.dataset_downloader import (
    HuggingFaceDatasetConfig,
    HuggingFaceDatasetDownloader,
)
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./config.yml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logger = logging.getLogger("download")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main(
    download_datasource: bool,
    count_tokens: bool,
    tokenizer_path: Path,
    dataset_configs: list[HuggingFaceDatasetConfig],
) -> None:
    logger = build_logger()

    if download_datasource:
        for config in dataset_configs:
            downloader = HuggingFaceDatasetDownloader(logger=logger)
            downloader.download_as_jsonl(config=config)

    if count_tokens:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.expanduser())
        for config in dataset_configs:
            dataset_dir = config.output_filepath.expanduser().parent
            logger.info(f"Processing dataset directory: {dataset_dir}")
            for filepath in dataset_dir.glob("**/*.jsonl"):
                output_filepath = (
                    filepath.parent / f"{filepath.stem}_with_token_count.jsonl"
                )
                with open(filepath, "r") as in_f, open(output_filepath, "a") as out_f:
                    for i, line in enumerate(in_f):
                        data = json.loads(line)
                        tokens = tokenizer.apply_chat_template(
                            json.loads(data["prompt"]),
                            add_generation_prompt=True,
                            tokenize=True,
                        )
                        data["token_counts"] = len(tokens)
                        data["sample_id"] = i
                        out_f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    download_datasource = config["download_datasource"]
    count_tokens = config["count_tokens"]
    tokenizer_path = Path(config["tokenizer_path"])
    dataset_configs = [
        HuggingFaceDatasetConfig.model_validate(config["dataset_configs"][key])
        for key in config["dataset_configs"]
    ]

    main(
        download_datasource=download_datasource,
        count_tokens=count_tokens,
        tokenizer_path=tokenizer_path,
        dataset_configs=dataset_configs,
    )
