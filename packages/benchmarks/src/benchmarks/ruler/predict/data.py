from typing import Any

from pydantic import model_validator

from ..._core.predict.data import Input, Output


class RULERInput(Input):
    prompt: str
    answer_prefix: str
    answer: list[str]
    needle_depth: list[float]

    # Optional (NIAH)
    needles: list[str] | None = None
    query: str | None = None

    # Optional (QA)
    question: str | None = None

    @model_validator(mode="before")
    def validate_content(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["prompt"] = values["content"]["user_prompt"]
        values["answer"] = values["content"]["outputs"]
        values["answer_prefix"] = values["content"]["answer_prefix"]
        values["needle_depth"] = values["target_depth_percent"]
        return values


class RULEROutput(Output):
    answer: list[str]
    needle_depth: list[float]
