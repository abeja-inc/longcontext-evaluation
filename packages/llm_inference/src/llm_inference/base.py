from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Literal, TypeVar

from .data import Conversation, Prompt, Response


InputType = TypeVar("InputType", Prompt, Conversation)


class BaseGenerator(ABC):
    def __init__(
        self,
        *,
        model_name: str,
        max_context_length: int,
        max_output_tokens: int,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens
        self.logger = logger
        self.tokenizer: Any
        self.tokenizer_type: Literal["huggingface", "tiktoken"]

    @property
    def default_too_long_input_error_message(self) -> str:
        return "[ERROR]: Input is too long."

    @abstractmethod
    def _count_tokens(self, input: Prompt | Conversation, **kwargs: Any) -> int: ...

    def _is_over_context_length(
        self,
        input: Prompt | Conversation,
        max_context_length: int,
        max_output_tokens: int,
        buffer_tokens: int,
        **kwargs: Any,
    ) -> bool:
        token_count: int = self._count_tokens(input=input, **kwargs)
        return token_count + max_output_tokens + buffer_tokens > max_context_length

    def _filter_long_inputs(
        self,
        inputs: list[InputType],
        max_context_length: int,
        max_output_tokens: int,
        buffer_tokens: int,
        **kwargs: Any,
    ) -> tuple[list[InputType], list[int]]:
        filtered_inputs: list[InputType] = []
        skip_idx: list[int] = []
        for index, input in enumerate(inputs):
            if self._is_over_context_length(
                input=input,
                max_context_length=max_context_length,
                max_output_tokens=max_output_tokens,
                buffer_tokens=buffer_tokens,
                **kwargs,
            ):
                skip_idx.append(index)
            else:
                filtered_inputs.append(input)
        return filtered_inputs, skip_idx

    @abstractmethod
    def chat(
        self, *, conversations: list[Conversation], **kwargs: Any
    ) -> list[Response]: ...

    @abstractmethod
    def completion(self, *, prompts: list[Prompt], **kwargs: Any) -> list[Response]: ...
