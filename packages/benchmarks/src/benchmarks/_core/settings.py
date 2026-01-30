from typing import Literal, TypeVar

from pydantic import BaseModel


class BaseSettings(BaseModel):
    use_truncate: bool
    truncate_type: Literal["middle"] = "middle"


SettingsType = TypeVar("SettingsType", bound=BaseSettings)
