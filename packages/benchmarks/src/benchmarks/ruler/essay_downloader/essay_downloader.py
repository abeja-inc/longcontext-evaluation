import json
import urllib.request
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import EssayConfig


class EssayDownloader:
    def __init__(self, logger: Logger):
        self.logger = logger

        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = True
        self.html_converter.escape_all = True  # pyright: ignore[reportAttributeAccessIssue]
        self.html_converter.reference_links = False  # pyright: ignore[reportAttributeAccessIssue]
        self.html_converter.mark_code = False

    def _load_urls(self, url_list_path: Path) -> list[str]:
        with url_list_path.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    def _download_html(self, url: str, tmp_dir: Path) -> None:
        filename = url.split("/")[-1].replace(".html", ".txt")
        output_path = tmp_dir / filename
        try:
            with urllib.request.urlopen(url) as website:
                content = website.read().decode("unicode_escape", "utf-8")
                soup = BeautifulSoup(content, "html.parser")
                tag = soup.find("font") or soup.body
                parsed = self.html_converter.handle(str(tag))
            output_path.write_text(parsed)
        except Exception as e:
            print(f"[HTML] Failed: {url} → {e}")

    def _download_text(self, url: str, tmp_dir: Path) -> None:
        filename = url.split("/")[-1]
        output_path = tmp_dir / filename
        try:
            with urllib.request.urlopen(url) as website:
                content = website.read().decode("utf-8")
            output_path.write_text(content)
        except Exception as e:
            print(f"[TXT] Failed: {url} → {e}")

    def _aggregate_text(self, files: list[Path]) -> str:
        content = ""
        for file_path in files:
            try:
                content += file_path.read_text()
            except Exception as e:
                print(f"[ReadError] {file_path} → {e}")
        return content

    def download(self, config: EssayConfig) -> None:
        with TemporaryDirectory(dir=config.output_filepath.parent) as tmp:
            tmp_dir = Path(tmp)

            tmp_html_dir = tmp_dir / "essay_html"
            tmp_repo_dir = tmp_dir / "essay_repo"
            tmp_html_dir.mkdir(parents=True, exist_ok=True)
            tmp_repo_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Downloading essays...")
            urls: list[str] = self._load_urls(url_list_path=config.url_list_path)
            for url in tqdm(urls, desc="Downloading", leave=False):
                if ".html" in url:
                    self._download_html(url=url, tmp_dir=tmp_html_dir)
                else:
                    self._download_text(url=url, tmp_dir=tmp_repo_dir)

            html_files: list[Path] = sorted(tmp_html_dir.glob("*.html"))
            self.logger.info("Downloaded %s HTML-style essays", len(html_files))
            repo_files: list[Path] = sorted(tmp_repo_dir.glob("*.txt"))
            self.logger.info("Downloaded %s GitHub-style essays", len(repo_files))

            all_text: str = self._aggregate_text(repo_files + html_files)
            config.output_filepath.parent.mkdir(parents=True, exist_ok=True)
            config.output_filepath.write_text(
                json.dumps({"text": all_text}, ensure_ascii=False)
            )
            self.logger.info("Saved combined essays")

        self.logger.info("Temporary directories cleaned up")
