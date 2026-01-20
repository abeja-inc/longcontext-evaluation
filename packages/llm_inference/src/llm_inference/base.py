from abc import ABC, abstractmethod
from typing import Any

from .data import Conversation, Prompt, Response


class BaseGenerator(ABC):
    @property
    def default_too_long_input_error_message(self) -> str:
        return "[ERROR]: Input is too long."

    @abstractmethod
    def _count_tokens(self, input: Prompt | Conversation) -> int: ...

    def _is_over_context_length(
        self,
        input: Prompt | Conversation,
        max_context_length: int,
        max_output_tokens: int,
        buffer_tokens: int,
    ) -> bool:
        token_count: int = self._count_tokens(input=input)
        return token_count + max_output_tokens + buffer_tokens > max_context_length

    def _filter_long_inputs(
        self,
        inputs: list[Prompt] | list[Conversation],
        max_context_length: int,
        max_output_tokens: int,
        buffer_tokens: int,
    ) -> tuple[list[Prompt] | list[Conversation], list[int]]:
        filtered_inputs: list[Prompt] | list[Conversation] = []
        skip_idx: list[int] = []
        for index, input in enumerate(inputs):
            if self._is_over_context_length(
                input=input,
                max_context_length=max_context_length,
                max_output_tokens=max_output_tokens,
                buffer_tokens=buffer_tokens,
            ):
                skip_idx.append(index)
            else:
                filtered_inputs.append(input)
        return filtered_inputs, skip_idx

    @abstractmethod
    def chat(
        self,
        *,
        conversations: list[Conversation],
        buffer_tokens: int = 10,
        **kwargs: Any,
    ) -> list[Response]: ...

    @abstractmethod
    def completion(
        self, *, prompts: list[Prompt], buffer_tokens: int = 10, **kwargs: Any
    ) -> list[Response]: ...
