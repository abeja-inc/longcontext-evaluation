from typing import TypeVar

from pydantic import AliasChoices, BaseModel, Field


class Input(BaseModel):
    id: str | int = Field(validation_alias=AliasChoices("id", "_id", "sample_id"))
    tokens: int = Field(
        validation_alias=AliasChoices(
            "tokens",
            "token_count",
            "token_counts",
            "target_token_count",
            "target_token_counts",
        )
    )


class Output(BaseModel):
    id: str | int
    input: str
    answer: object
    output: str
    context_length: int

    output_reasoning: str | None = None


OutputType = TypeVar("OutputType", bound=Output)
