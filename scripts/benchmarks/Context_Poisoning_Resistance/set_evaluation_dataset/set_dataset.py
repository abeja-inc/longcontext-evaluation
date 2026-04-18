import argparse
import json
import logging
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


def _rotate_numbers(numbers: list[str], shift: int) -> list[str]:
    shift = shift % len(numbers)
    return numbers[shift:] + numbers[:shift]


def _build_orders(raw_puzzles: list[dict[str, str]]) -> dict[str, list[str]]:
    orders: dict[str, list[str]] = {}
    for idx, puzzle in enumerate(raw_puzzles):
        numbers = [
            puzzle["num_1"],
            puzzle["num_2"],
            puzzle["num_3"],
            puzzle["num_4"],
        ]
        orders[puzzle["id"]] = _rotate_numbers(numbers, idx)
    return orders


def _write_condition_dataset(
    *,
    raw_puzzles: list[dict[str, str]],
    number_orders: dict[str, list[str]],
    output_filepath: Path,
    condition: str,
    judge_label: str,
    support_length: int,
) -> None:
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with output_filepath.open("w", encoding="utf-8") as f:
        for target_idx, puzzle in enumerate(raw_puzzles):
            target_id = puzzle["id"]
            support_ids = [
                raw_puzzles[(target_idx + offset + 1) % len(raw_puzzles)]["id"]
                for offset in range(support_length)
            ]
            record = {
                "id": f"target-{target_id}__{condition}__k{support_length}",
                "sample_id": f"target-{target_id}__{condition}__k{support_length}",
                "target_id": target_id,
                "support_ids": support_ids,
                "condition": condition,
                "language": "japanese",
                "target_numbers": number_orders[target_id],
                "support_number_orders": {
                    support_id: number_orders[support_id] for support_id in support_ids
                },
                "judge_label": judge_label,
                "tokens": 0,
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def main(source_filepath: Path, output_dir: Path, support_lengths: list[int]) -> None:
    logger = build_logger()
    raw_puzzles = [
        json.loads(line)
        for line in source_filepath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    number_orders = _build_orders(raw_puzzles)
    max_support_length = len(raw_puzzles) - 1

    logger.info("Writing evaluation datasets to %s", output_dir)
    for support_length in support_lengths:
        if support_length > max_support_length:
            logger.warning(
                "Skip support_length=%d because max available support length is %d",
                support_length,
                max_support_length,
            )
            continue

        length_dir = output_dir / f"k{support_length}"
        _write_condition_dataset(
            raw_puzzles=raw_puzzles,
            number_orders=number_orders,
            output_filepath=length_dir / "clean.jsonl",
            condition="clean",
            judge_label="correct",
            support_length=support_length,
        )
        _write_condition_dataset(
            raw_puzzles=raw_puzzles,
            number_orders=number_orders,
            output_filepath=length_dir / "poisoned.jsonl",
            condition="poisoned",
            judge_label="correct",
            support_length=support_length,
        )
        _write_condition_dataset(
            raw_puzzles=raw_puzzles,
            number_orders=number_orders,
            output_filepath=length_dir / "poisoned_marked_incorrect.jsonl",
            condition="poisoned_marked_incorrect",
            judge_label="incorrect",
            support_length=support_length,
        )


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(
        source_filepath=Path(config["source_filepath"]),
        output_dir=Path(config["output_dir"]),
        support_lengths=[int(length) for length in config["support_lengths"]],
    )
