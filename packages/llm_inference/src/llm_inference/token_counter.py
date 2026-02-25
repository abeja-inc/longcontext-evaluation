from typing import Any, Literal

import tiktoken
from transformers import AutoTokenizer

from .data import Conversation, Prompt


class TokenCounter:
    """Tokenizer-backed utility for unified token counting.

    This class provides one canonical counting path for `Prompt` and `Conversation`
    inputs across inference and benchmark code.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        tokenizer_type: Literal["huggingface", "tiktoken"],
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer_type = tokenizer_type
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: Any
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

    @classmethod
    def from_tokenizer(
        cls,
        *,
        tokenizer: Any,
        tokenizer_type: Literal["huggingface", "tiktoken"],
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> "TokenCounter":
        """Build a token counter from an already initialized tokenizer instance."""
        instance = cls.__new__(cls)
        instance.tokenizer: Any = tokenizer
        instance.tokenizer_type = tokenizer_type
        instance.tokenizer_kwargs = tokenizer_kwargs or {}
        return instance

    def count_prompt_tokens(self, prompt: Prompt | str) -> int:
        """Count tokens for a completion-style prompt."""
        text = prompt.prompt if isinstance(prompt, Prompt) else prompt
        if self.tokenizer_type == "huggingface":
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def count_conversation_tokens(
        self,
        conversation: Conversation,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool | None = None,
    ) -> int:
        """Count tokens for a chat-style conversation.

        Args:
            conversation: Conversation to count.
            chat_template_kwargs: Additional kwargs for HF `apply_chat_template`.
            add_generation_prompt: Whether to include generation prompt tokens when
                applying HF chat templates. If omitted, HF defaults are used.
                For tiktoken, this has no effect because counting follows the OpenAI
                message accounting formula.
        """
        if self.tokenizer_type == "huggingface":
            template_kwargs = dict(chat_template_kwargs or {})
            if add_generation_prompt is not None:
                template_kwargs["add_generation_prompt"] = add_generation_prompt
            return len(
                self.tokenizer.apply_chat_template(
                    conversation.prompt,
                    tokenize=True,
                    **template_kwargs,
                )
            )

        # Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in conversation.prompt:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                num_tokens += len(self.tokenizer.encode(value, disallowed_special=()))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3
        return num_tokens

    def count_tokens(
        self,
        input: str | Prompt | Conversation,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool | None = None,
    ) -> int:
        """Count tokens for plain text, prompt, or conversation input."""
        if isinstance(input, Conversation):
            return self.count_conversation_tokens(
                input,
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
        return self.count_prompt_tokens(input)
