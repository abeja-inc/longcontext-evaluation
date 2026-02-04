from typing import Literal, TypeVar

from pydantic import BaseModel


class BaseSettings(BaseModel):
    use_truncate: bool
    truncate_type: Literal["middle"] | None = "middle"
    truncate_buffer_tokens: int = 10


SettingsType = TypeVar("SettingsType", bound=BaseSettings)
