from logging import Logger
from typing import Any

from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse

from .base import BaseGenerator
from .data import Conversation, OutputContent, Prompt, Response


class OpenAIGenerator(BaseGenerator):
    def __init__(
        self,
        *,
        model: str,
        max_context_length: int,
        max_output_tokens: int,
        logger: Logger,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens
        self.logger = logger
        self.client = OpenAI(**kwargs)

    def _call_token_count_api(self, input: Prompt | Conversation) -> int:
        response = self.client.responses.input_tokens.count(  # pyright: ignore[reportUnknownVariableType]
            model=self.model,
            input=input.prompt,  # pyright: ignore[reportArgumentType]
        )
        return response.input_tokens

    def _count_tokens(self, input: Prompt | Conversation) -> int:
        return self._call_token_count_api(input=input)

    def _call_response_api(
        self,
        *,
        inputs: list[Prompt] | list[Conversation],
        buffer_tokens: int = 10,
        **kwargs: Any,
    ) -> list[Response]:
        responses: list[Response] = []
        for input in inputs:
            if self._is_over_context_length(
                input=input,
                max_context_length=self.max_context_length,
                max_output_tokens=self.max_output_tokens,
                buffer_tokens=buffer_tokens,
            ):
                responses.append(
                    Response(
                        input=input.prompt,
                        outputs=[
                            OutputContent(
                                content=self.default_too_long_input_error_message,
                                reasoning_content=None,
                            )
                        ],
                        metadata=input.metadata,
                    )
                )
            else:
                api_response: OpenAIResponse = self.client.responses.create(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
                    model=self.model,
                    input=input.prompt,  # pyright: ignore[reportArgumentType]
                    max_output_tokens=self.max_output_tokens,
                    **kwargs,
                )

                outputs: list[OutputContent] = [
                    OutputContent(
                        content=api_response.output_text,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        reasoning_content=None,
                    )
                ]

                metadata: dict[str, Any] = input.metadata or {}
                if api_response.usage:  # pyright: ignore[reportUnknownMemberType]
                    metadata["usage"] = api_response.usage.model_dump()  # pyright: ignore[reportUnknownMemberType]

                responses.append(
                    Response(
                        input=input.prompt,
                        outputs=outputs,
                        metadata=metadata if metadata else None,
                    )
                )
        return responses

    def chat(
        self,
        *,
        conversations: list[Conversation],
        buffer_tokens: int = 10,
        **kwargs: Any,
    ) -> list[Response]:
        return self._call_response_api(
            inputs=conversations, buffer_tokens=buffer_tokens, **kwargs
        )

    def completion(
        self, *, prompts: list[Prompt], buffer_tokens: int = 10, **kwargs: Any
    ) -> list[Response]:
        return self._call_response_api(
            inputs=prompts, buffer_tokens=buffer_tokens, **kwargs
        )
