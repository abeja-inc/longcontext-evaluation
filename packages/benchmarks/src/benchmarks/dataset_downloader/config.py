from pathlib import Path

from pydantic import BaseModel


class DownloadConfig(BaseModel):
    output_filepath: Path


class HuggingFaceDatasetConfig(DownloadConfig):
    huggingface_dataset_name: str
    name: str | None = None
    split: str | None = None
    trust_remote_code: bool | None = None
    data_files: dict[str, str | list[str]] | None = None


class URLConfig(DownloadConfig):
    dataset_url: str
