from typing import Literal, TypeVar

from pydantic import BaseModel


class BaseSettings(BaseModel):
    use_truncate: bool
    truncate_type: Literal["middle", "last_n_turns"] = "middle"
    truncate_buffer_tokens: int = 10
    require_reasoning: bool = False


SettingsType = TypeVar("SettingsType", bound=BaseSettings)
