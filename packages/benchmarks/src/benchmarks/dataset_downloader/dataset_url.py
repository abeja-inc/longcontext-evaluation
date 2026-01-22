from logging import Logger

import requests

from .config import URLConfig


class URLDownloader:
    def __init__(self, logger: Logger):
        self.logger = logger

    def download(self, config: URLConfig) -> None:
        response = requests.get(config.dataset_url)
        response.raise_for_status()
        text = response.text
        self.logger.info("Downloaded dataset from %s", config.dataset_url)

        config.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        config.output_filepath.write_text(text, encoding="utf-8")
        self.logger.info("Saved dataset to %s", config.output_filepath)
