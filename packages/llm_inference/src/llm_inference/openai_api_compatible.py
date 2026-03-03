from logging import Logger

from openai import OpenAI
from transformers import AutoTokenizer

from .openai_api import OpenAIGenerator
from .token_counter import TokenCounter


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
        self.tokenizer_type = "huggingface"
        self.token_counter = TokenCounter.from_tokenizer(
            tokenizer=self.tokenizer, tokenizer_type=self.tokenizer_type
        )
