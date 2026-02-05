from typing import Any

from pydantic import model_validator

from ..._core.predict.data import Input, Output


class RULERInput(Input):
    prompt: str
    answer_prefix: str
    answer: list[str]
    target_depth_percent: float

    # Optional (NIAH)
    needles: list[str] | None = None
    query: str | None = None

    # Optional (QA)
    question: str | None = None

    @model_validator(mode="before")
    def validate_content(cls, values: dict[str, Any]) -> dict[str, Any]:
        content = values.get("content", {})
        values["prompt"] = content.get("prompt", "")
        values["answer"] = content.get("outputs", [])
        values["answer_prefix"] = content.get("answer_prefix", "")
        return values


class RULEROutput(Output):
    needle_depth: float
