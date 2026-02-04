import json
from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: list[Message]
    metadata: dict[str, Any] | None = None

    @property
    def prompt(self) -> list[dict[str, str]]:
        return [message.model_dump() for message in self.messages]

    @property
    def to_string(self) -> str:
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in self.messages)


class Prompt(BaseModel):
    prompt: str
    metadata: dict[str, Any] | None = None

    @property
    def to_string(self) -> str:
        return self.prompt


class OutputContent(BaseModel):
    content: str
    reasoning_content: str | None = None


class Response(BaseModel):
    input: str | list[dict[str, str]]
    outputs: list[OutputContent]
    metadata: dict[str, Any] | None = None
