from pathlib import Path

from pydantic import BaseModel

from .._core.settings import BaseSettings


class PromptFiles(BaseModel):
    zero_shot: str
    zero_shot_cot: str
    zero_shot_cot_ans: str
    zero_shot_no_context: str
    zero_shot_rag: str


class Prompt(Path):
    dirpath: Path
    files: PromptFiles


class LongBenchV2Settings(BaseSettings):
    rag_topn: int
    cot: bool
    no_context: bool
    compensate_missing: bool
    prompt: Prompt
