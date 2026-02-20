import argparse
import logging
from pathlib import Path

import yaml

from benchmarks.dataset_downloader import (
    HuggingFaceDatasetConfig,
    HuggingFaceDatasetDownloader,
    URLConfig,
    URLDownloader,
)
from benchmarks.ruler.essay_downloader import EssayConfig, EssayDownloader


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
    hf_dataset_configs: list[HuggingFaceDatasetConfig],
    url_dataset_configs: list[URLConfig],
    essay_config: EssayConfig,
) -> None:
    logegr = build_logger()

    if download_datasource:
        essay_downloader = EssayDownloader(logger=logegr)
        essay_downloader.download(config=essay_config)

        for config in hf_dataset_configs:
            downloader = HuggingFaceDatasetDownloader(logger=logegr)
            downloader.download_as_jsonl(config=config)

        for config in url_dataset_configs:
            downloader = URLDownloader(logger=logegr)
            downloader.download(config=config)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    download_datasource = config["download_datasource"]
    hf_dataset_configs = [
        HuggingFaceDatasetConfig.model_validate(config["hf_dataset_configs"][key])
        for key in config["hf_dataset_configs"]
    ]
    url_dataset_configs = [
        URLConfig.model_validate(config["url_dataset_configs"][key])
        for key in config["url_dataset_configs"]
    ]
    essay_config = EssayConfig.model_validate(config["essay_dataset_config"])

    main(
        download_datasource=download_datasource,
        hf_dataset_configs=hf_dataset_configs,
        url_dataset_configs=url_dataset_configs,
        essay_config=essay_config,
    )
