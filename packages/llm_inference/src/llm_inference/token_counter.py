from typing import Any, Literal

import tiktoken
from transformers import AutoTokenizer

from .data import Conversation, Prompt


class TokenCounter:
    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        tokenizer_type: Literal["huggingface", "tiktoken"],
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer_type = tokenizer_type
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        if tokenizer_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                **self.tokenizer_kwargs,
            )
        elif tokenizer_type == "tiktoken":
            try:
                self.tokenizer = tiktoken.encoding_for_model(tokenizer_name_or_path)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding(tokenizer_name_or_path)
        else:
            raise ValueError("Supported tokenizer_type: huggingface or tiktoken")

    def count_tokens(self, input: str | Prompt | Conversation, **kwargs: Any) -> int:
        if self.tokenizer_type == "huggingface":
            if isinstance(input, Conversation):
                return len(
                    self.tokenizer.apply_chat_template(
                        input.prompt, tokenize=True, **kwargs
                    )
                )
            if isinstance(input, Prompt):
                return len(
                    self.tokenizer.encode(input.prompt, add_special_tokens=False)
                )
            return len(self.tokenizer.encode(input, add_special_tokens=False))

        if isinstance(input, Prompt):
            return len(self.tokenizer.encode(input.prompt, disallowed_special=()))
        if isinstance(input, Conversation):
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
                    num_tokens += len(
                        self.tokenizer.encode(value, disallowed_special=())
                    )
                    if key == "name":
                        num_tokens += tokens_per_name

            num_tokens += 3
            return num_tokens
        return len(self.tokenizer.encode(input, disallowed_special=()))
