from pathlib import Path
from string import Template
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .._core.settings import BaseSettings


class PromptFiles(BaseModel):
    system_prompt: str
    question: str
    question_prefix: str


class Prompt(BaseModel):
    dirpath: Path
    files: PromptFiles

    @model_validator(mode="after")
    def _check_files_exist(self) -> "Prompt":
        for filename in self.files.model_dump().values():
            path = self.dirpath / filename
            if not path.is_file():
                raise ValueError(f"Prompt template file is not found: {path}")
        return self

    def load_templates(self) -> dict[str, Template]:
        return {
            key: Template((self.dirpath / filename).read_text(encoding="utf-8"))
            for key, filename in self.files.model_dump().items()
        }


class ContextPoisoningResistanceSettings(BaseSettings):
    use_truncate: bool = False
    truncate_type: Literal["middle", "last_n_turns"] = "last_n_turns"
    prompt: Prompt
    stage1_cache_filename: str = "_stage1_cache.jsonl"
    metric_kwargs: dict[str, Any] = Field(default_factory=dict)
