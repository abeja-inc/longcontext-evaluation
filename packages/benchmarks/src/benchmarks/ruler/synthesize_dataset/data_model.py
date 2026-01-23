from pydantic import BaseModel, Field


class Content(BaseModel):
    user_prompt: str
    answer_prefix: str
    outputs: list[str]


class BaseDatasetSchema(BaseModel):
    content: Content
    target_context_length: int = Field(
        description="The target context length category for the test (e.g., 4096, 8192)."
    )
    target_depth_percent: list[float] = Field(
        description="The percentage depths at which the needle is placed in the context."
    )
    input_content_length: int = Field(
        description="The actual number of tokens in the input context."
    )
    input_length_w_model_temp: int = Field(
        description="The actual number of tokens in the input prompt."
    )
    task: str
    subset: str
    sample_id: int


class NIAHDatasetSchema(BaseDatasetSchema):
    needles: list[str] = Field(
        description="List of all needle strings inserted into the context."
    )
    query: str = Field(description="The final query string asking about specific keys.")
    answer_prefix: str | None = Field(
        default=None, description="The prefix of the answer part in the prompt."
    )


class QADatasetSchema(BaseDatasetSchema):
    question: str
    target_context: str


class QAPair(BaseModel):
    question: str
    answers: list[str]
    context_indices: list[int]
    more_context_indices: list[int] | None


class ProcessedQAData(BaseModel):
    contexts: list[str]
    qas: list[QAPair]
