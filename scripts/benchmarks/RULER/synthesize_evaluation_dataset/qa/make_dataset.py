import argparse
import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml
from benchmarks.ruler.synthesize_dataset import (
    HotpotQAGenerator,
    JEMHopQAGenerator,
    JSQuADGenerator,
    QASynthesisConfig,
    SQuADGenerator,
)


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], Mapping) and isinstance(v, Mapping):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def build_logger() -> logging.Logger:
    logger = logging.getLogger("synthesize dataset")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("./config.yaml"),
        help="Path to YAML jobs file",
    )
    return p.parse_args()


def main(synth_configs: list[QASynthesisConfig], logger: logging.Logger) -> None:
    for i, config in enumerate(synth_configs):
        logger.info(
            "Synthesize dataset %s/%s: '%s' ---",
            i + 1,
            len(synth_configs),
            config.subset,
        )

        try:
            if config.subset == "squad":
                generator = SQuADGenerator(config, logger=logger)
            elif config.subset == "hotpotqa":
                generator = HotpotQAGenerator(config, logger=logger)
            elif config.subset == "jsquad":
                generator = JSQuADGenerator(config, logger=logger)
            elif config.subset == "jemhopqa":
                generator = JEMHopQAGenerator(config, logger=logger)
            else:
                raise ValueError(
                    f"No generator available for dataset '{config.subset}'"
                )

            generator.run()
            logger.info("Successfully synthesized.")

        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info("Complete all jobs.")


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config["dataset_root"])
    base_config = config["base_config"]
    synth_configs = [
        QASynthesisConfig(
            **deep_merge(base=base_config, override=subset_config["config_override"]),
            subset=subset_config["name"],
            save_dirpath=dataset_root,
            qa_dataset_path=dataset_root / subset_config["source_dataset_file"],
        )
        for subset_config in config["subset_configs"]
    ]

    logger = build_logger()

    main(synth_configs=synth_configs, logger=logger)
