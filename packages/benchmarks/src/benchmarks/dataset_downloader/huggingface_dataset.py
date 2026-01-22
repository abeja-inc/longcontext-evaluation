import json
from logging import Logger
from pathlib import Path

from tqdm import tqdm

from datasets import (
    Dataset,
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)

from .config import HuggingFaceDatasetConfig


class HuggingFaceDatasetDownloader:
    def __init__(self, logger: Logger):
        self.logger = logger

    def _save_dataset_as_jsonl(self, dataset: Dataset, output_filepath: Path):
        self.logger.info("Saving dataset to %s", output_filepath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with output_filepath.open("w", encoding="utf-8") as f:
            for record in tqdm(dataset, desc="Writing JSONL"):  # pyright: ignore[reportUnknownVariableType]
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        self.logger.info("Dataset saved: %d records", len(dataset))

    def download_as_jsonl(self, config: HuggingFaceDatasetConfig):
        dataset = load_dataset(
            path=str(config.huggingface_dataset_name),
            name=config.name,
            split=config.split,
            data_files=config.data_files,
            trust_remote_code=config.trust_remote_code,
            streaming=False,
        )

        if isinstance(dataset, Dataset):
            self.logger.info("Loaded Dataset Info:\n%s", dataset.info)
            self._save_dataset_as_jsonl(
                dataset=dataset, output_filepath=config.output_filepath
            )
        else:
            for split_name in dataset.keys():
                self.logger.info("Loaded Dataset Info:\n%s", dataset[split_name].info)

                output_filepath = (
                    config.output_filepath.parent
                    / str(split_name)
                    / f"{config.output_filepath.name}"
                )
                self._save_dataset_as_jsonl(
                    dataset=dataset[split_name], output_filepath=output_filepath
                )
