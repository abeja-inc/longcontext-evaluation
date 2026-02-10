import argparse
import logging
import shutil
from pathlib import Path

import yaml


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


def main(source_filepath: Path, output_filepath: Path) -> None:
    logger = build_logger()

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {source_filepath} to {output_filepath}")
    shutil.copy(source_filepath, output_filepath)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(
        source_filepath=Path(config["source_filepath"]),
        output_filepath=Path(config["output_filepath"]),
    )
