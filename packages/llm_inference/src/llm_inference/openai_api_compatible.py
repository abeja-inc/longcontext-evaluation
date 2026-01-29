from logging import Logger
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

from .data import Conversation, Prompt
from .openai_api import OpenAIGenerator


class OpenAICompatibleGenerator(OpenAIGenerator):
    def __init__(
        self,
        *,
        client: OpenAI,
        tokenizer: AutoTokenizer,
        model_name: str,
        max_context_length: int,
        max_output_tokens: int,
        logger: Logger,
    ) -> None:
        super().__init__(
            client=client,
            model_name=model_name,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            logger=logger,
        )
        self.tokenizer = tokenizer

    def _call_token_count_api(self, input: Prompt | Conversation, **kwargs: Any) -> int:
        if isinstance(input, Conversation):
            return len(
                self.tokenizer.apply_chat_template(
                    input.prompt, tokenize=True, **kwargs
                )
            )
        else:
            return len(self.tokenizer.encode(input.prompt, add_special_tokens=False))
