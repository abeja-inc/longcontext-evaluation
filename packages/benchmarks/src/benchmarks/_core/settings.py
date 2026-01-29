from typing import TypeVar

from pydantic import BaseModel


class BaseSettings(BaseModel):
    use_truncate: bool


SettingsType = TypeVar("SettingsType", bound=BaseSettings)
