from logging import Logger
from typing import Any

import tiktoken
from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse

from .base import BaseGenerator
from .data import Conversation, OutputContent, Prompt, Response


class OpenAIGenerator(BaseGenerator):
    def __init__(
        self,
        *,
        client: OpenAI,
        model_name: str,
        max_context_length: int,
        max_output_tokens: int,
        logger: Logger,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            logger=logger,
        )
        self.client = client
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.tokenizer_type = "tiktoken"

    def _count_tokens(self, input: Prompt | Conversation, **kwargs: Any) -> int:
        if isinstance(input, Prompt):
            return len(self.tokenizer.encode(input.prompt))
        elif isinstance(input, Conversation):
            # Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            tokens_per_message = 3
            tokens_per_name = 1

            num_tokens = 0
            for message in input.prompt:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        value = str(value)
                    num_tokens += len(self.tokenizer.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

            num_tokens += 3
            return num_tokens
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

    def _call_response_api(
        self,
        *,
        inputs: list[Prompt] | list[Conversation],
        long_input_filter_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Response]:
        responses: list[Response] = []
        for input in inputs:
            if self._is_over_context_length(
                input=input,
                max_context_length=self.max_context_length,
                max_output_tokens=self.max_output_tokens,
                **long_input_filter_kwargs,
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
                api_response: OpenAIResponse = self.client.responses.create(
                    model=self.model_name,
                    input=input.prompt,
                    max_output_tokens=self.max_output_tokens,
                    **kwargs,
                )

                outputs: list[OutputContent] = [
                    OutputContent(
                        content=api_response.output_text, reasoning_content=None
                    )
                ]

                metadata: dict[str, Any] = input.metadata or {}
                if api_response.usage:
                    metadata["usage"] = api_response.usage.model_dump()

                responses.append(
                    Response(
                        input=input.prompt,
                        outputs=outputs,
                        metadata=metadata if metadata else None,
                    )
                )
        return responses

    def _chat(
        self,
        *,
        conversations: list[Conversation],
        long_input_filter_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Response]:
        return self._call_response_api(
            inputs=conversations,
            long_input_filter_kwargs=long_input_filter_kwargs,
            **kwargs,
        )

    def _completion(
        self,
        *,
        prompts: list[Prompt],
        long_input_filter_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Response]:
        return self._call_response_api(
            inputs=prompts, long_input_filter_kwargs=long_input_filter_kwargs, **kwargs
        )
