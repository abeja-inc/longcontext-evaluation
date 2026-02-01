from pathlib import Path
from string import Template
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .._core.settings import BaseSettings


class PromptFiles(BaseModel):
    zero_shot: str
    zero_shot_cot: str
    zero_shot_cot_ans: str
    zero_shot_no_context: str
    zero_shot_rag: str


class Prompt(BaseModel):
    dirpath: Path
    files: PromptFiles

    @model_validator(mode="after")
    def _check_files_exist(self) -> "Prompt":
        for key, filename in self.files.model_dump().items():
            path = self.dirpath / filename
            if not path.is_file():
                raise ValueError(
                    f"The prompt template file is not found: {key} -> {path}"
                )
        return self

    def load_templates(self) -> dict[str, Template]:
        out: dict[str, Template] = {}
        for key, filename in self.files.model_dump().items():
            path = self.dirpath / filename
            out[key] = Template(path.read_text(encoding="utf-8"))
        return out


class LongBenchV2Settings(BaseSettings):
    rag_topn: int
    cot: bool
    no_context: bool
    compensate_missing: bool
    prompt: Prompt
    metric_kwargs: dict[str, Any] = Field(default_factory=dict)
