from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BaseSynthesisConfig(BaseModel):
    save_dirpath: Path
    task: str  # タスク名
    subset: str  # サブセット名

    hf_tokenizer_path: str = "Qwen/Qwen3-4B"
    max_new_tokens: int = 32  # LLM推論時の生成トークン長
    apply_chat_template_kwargs: dict[str, Any] = {"enable_thinking": True}
    task_prompt_template: str

    context_lengths: list[int] = [128, 256, 512]  # テストするコンテキスト長のリスト
    num_samples: int = 10  # 各コンテキスト長でのテストのサンプル数
    random_seed: int = 42

    retry_limit: int = 10  # サンプル生成時のリトライ回数
    min_units: int = 2  # リトライ時の最小ユニット数
    initial_units: int = 8  # 指数関数的探索フェーズの初期ユニット数

    remove_newline_tab: bool = Field(
        default=False,
        description="Whether to remove newline and tab characters from the context.",
    )
    answer_prefix_template: str

    @property
    def prompt_dirpath(self) -> Path:
        return Path(__file__).parent.parent.parent / "prompts"


class NIAHSynthesisConfig(BaseSynthesisConfig):
    paulgraham_essay_path: Path  # Paul GrahamのエッセイのJSONファイルパス

    task_prompt_template: str = Field(
        default=(
            "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text? "
        ),
        description="Prompt template for the NIAH task.",
    )
    answer_prefix_template: str = "The special magic {type_needle_v} for {query} mentioned in the provided text are"

    # NIAHタスク固有のパラメータ
    num_needle_k: int = Field(
        default=1, description="Number of unique keys (topics) for needles."
    )
    num_needle_v: int = Field(
        default=1, description="Number of values (needles) per key."
    )
    num_needle_q: int = Field(default=1, description="Number of keys to query for.")

    type_haystack: Literal["essay", "noise", "needle"] = Field(
        default="essay", description="Type of text to use as haystack."
    )
    type_needle_k: Literal["numbers", "words", "uuids"] = Field(
        default="words", description="Data type for needle keys."
    )
    type_needle_v: Literal["numbers", "words", "uuids"] = Field(
        default="numbers", description="Data type for needle values."
    )

    needle_format: str = (
        "One of the special magic {type_needle_v} for {key} is: {value}."
    )

    @model_validator(mode="after")
    def _ensure_query_keys_within_total(self) -> "NIAHSynthesisConfig":
        if self.num_needle_k < self.num_needle_q:
            self.num_needle_k = self.num_needle_q
        return self


class QASynthesisConfig(BaseSynthesisConfig):
    qa_dataset_name: Literal["squad", "hotpotqa", "jsquad", "jemhopqa"] = Field(
        description="Name of the source QA dataset."
    )
    qa_dataset_path: Path = Field(
        description="Path to the source QA dataset JSON file."
    )
    task_prompt_template: str = Field(
        default=(
            "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
            "The following are given documents.\n\n{context}\n\n"
            "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
            "Question: {query}"
        ),
        description="Prompt template for the QA task.",
    )
    answer_prefix_template: str = "Answer:"
    document_prompt: str = "Document {i}:\n{document}"

    remove_newline_tab: bool = Field(
        default=False, description="Whether to remove newline and tab characters."
    )
    pre_samples: int = Field(
        default=0,
        description="Number of samples that are already generated to offset the starting point.",
    )
