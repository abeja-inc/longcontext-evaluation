from pathlib import Path

from ...dataset_downloader import DownloadConfig


class EssayConfig(DownloadConfig):
    url_list_path: Path
