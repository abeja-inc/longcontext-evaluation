import argparse
import logging
from pathlib import Path

import yaml
from benchmarks.dataset_downloader import (
    HuggingFaceDatasetConfig,
    HuggingFaceDatasetDownloader,
)


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
    download_datasource: bool, dataset_configs: list[HuggingFaceDatasetConfig]
) -> None:
    logegr = build_logger()

    if download_datasource:
        for config in dataset_configs:
            downloader = HuggingFaceDatasetDownloader(logger=logegr)
            downloader.download_as_jsonl(config=config)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    download_datasource = config["download_datasource"]
    dataset_configs = [
        HuggingFaceDatasetConfig.model_validate(config["dataset_configs"][key])
        for key in config["dataset_configs"]
    ]

    main(
        download_datasource=download_datasource,
        dataset_configs=dataset_configs,
    )
