import warnings
from typing import Any, Literal

from pydantic import Field, field_validator

from .._core.settings import BaseSettings


class OpenAIMRCRSettings(BaseSettings):
    use_truncate: bool = False
    truncate_type: Literal["middle", "last_n_turns"] = "last_n_turns"
    metric_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("use_truncate")
    def _check_use_truncate(cls, value: bool) -> bool:
        if value:
            warnings.warn(
                "Warning: truncation is not recommended for the benchmark OpenAI-MRCR."
            )
        return value

    @field_validator("truncate_type")
    def _check_truncate_type(
        cls, value: Literal["middle", "last_n_turns"]
    ) -> Literal["last_n_turns"]:
        if value not in ["middle", "last_n_turns"]:
            raise ValueError("Invalid truncate type")

        if value == "middle":
            raise ValueError(
                "Unsupported configuration: truncate type 'middle' is not recommended for the benchmark OpenAI-MRCR."
            )
        return value
