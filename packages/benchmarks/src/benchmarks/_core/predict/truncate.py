import warnings
from typing import Any, Literal, Protocol, cast

from llm_inference.data import Conversation
from llm_inference.token_counter import TokenCounter


class _HFTextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = ...
    ) -> str: ...


class _TikTokenTokenizer(Protocol):
    def encode(
        self, text: str, *, disallowed_special: tuple[str, ...]
    ) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


def _build_compat_token_counter(
    *,
    token_counter: TokenCounter | None,
    tokenizer: Any,
    tokenizer_type: Literal["huggingface", "tiktoken"] | None,
) -> TokenCounter:
    """Return provided token_counter, or construct one from legacy tokenizer args."""
    if token_counter is not None:
        return token_counter
    if tokenizer is None or tokenizer_type is None:
        raise ValueError(
            "Either token_counter must be provided, or both tokenizer and tokenizer_type are required"
        )
    return TokenCounter.from_tokenizer(
        tokenizer=tokenizer, tokenizer_type=tokenizer_type
    )


def _truncate_middle(
    text: str,
    *,
    token_counter: TokenCounter,
    max_context_length: int,
    max_output_tokens: int,
    buffer_tokens: int = 10,
) -> str:
    tokenizer_type = token_counter.tokenizer_type
    if tokenizer_type == "huggingface":
        tokenizer = cast(_HFTextTokenizer, token_counter.tokenizer)
        input_ids = tokenizer.encode(text)
        input_length = len(input_ids)
        if input_length + max_output_tokens > max_context_length:
            max_len = max_context_length - max_output_tokens - buffer_tokens
            truncated_input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            return tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
        return text

    if tokenizer_type == "tiktoken":
        tokenizer = cast(_TikTokenTokenizer, token_counter.tokenizer)
        input_ids = tokenizer.encode(text, disallowed_special=())
        input_length = len(input_ids)
        if input_length + max_output_tokens > max_context_length:
            max_len = max_context_length - max_output_tokens - buffer_tokens
            truncated_input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            return tokenizer.decode(truncated_input_ids)
        return text

    raise NotImplementedError("Supported tokenizer_type: huggingface or tiktoken")


def _truncate_last_n_turns(
    conversation: Conversation,
    *,
    token_counter: TokenCounter,
    max_context_length: int,
    max_output_tokens: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    add_generation_prompt: bool = True,
    buffer_tokens: int = 10,
) -> Conversation:
    messages = conversation.messages
    if not messages:
        raise ValueError("Conversation.messages is empty")

    budget = max_context_length - max_output_tokens - buffer_tokens
    if budget <= 0:
        raise ValueError(
            "Invalid budget: max_context_length is too small vs max_output_tokens/buffer_tokens"
        )

    user_idxs = [i for i, m in enumerate(messages) if m.role == "user"]
    if not user_idxs:
        raise ValueError("No user message found in conversation")

    first_user: int = user_idxs[0]
    last_user: int = user_idxs[-1]

    if first_user == last_user:
        if (
            token_counter.count_conversation_tokens(
                conversation,
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
            > budget
        ):
            warnings.warn(
                "Even the single-user conversation exceeds the context budget"
            )
        return conversation

    prefix_indices: list[int] = list(range(0, first_user))
    second_user: int = user_idxs[1]
    first_segment_indices: list[int] = list(range(first_user, second_user))
    last_user_indices: list[int] = [last_user]

    middle_segments: list[list[int]] = []
    for k in range(1, len(user_idxs) - 1):
        start = user_idxs[k]
        end = user_idxs[k + 1]
        middle_segments.append(list(range(start, end)))

    kept_indices: list[int] = []
    kept_indices.extend(prefix_indices)
    kept_indices.extend(first_segment_indices)
    kept_indices.extend(last_user_indices)
    kept_indices = sorted(set(kept_indices))

    mandatory_conv = Conversation(
        messages=[messages[i] for i in kept_indices],
        metadata=conversation.metadata,
    )
    if (
        token_counter.count_conversation_tokens(
            mandatory_conv,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
        )
        > budget
    ):
        warnings.warn("Even mandatory messages exceed the context budget")
        return conversation

    kept_middle_count = len(middle_segments)
    while True:
        kept_indices = []
        kept_indices.extend(prefix_indices)
        kept_indices.extend(first_segment_indices)

        for seg in middle_segments[:kept_middle_count]:
            kept_indices.extend(seg)

        kept_indices.extend(last_user_indices)
        kept_indices = sorted(set(kept_indices))

        truncated_conversation = Conversation(
            messages=[messages[i] for i in kept_indices],
            metadata=conversation.metadata,
        )

        if (
            token_counter.count_conversation_tokens(
                truncated_conversation,
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
            )
            <= budget
        ):
            return truncated_conversation

        if kept_middle_count == 0:
            warnings.warn("Cannot truncate to fit the context budget")
            return truncated_conversation

        kept_middle_count -= 1


def truncate_text(
    text: str | Conversation,
    tokenizer: Any = None,
    max_context_length: int = 0,
    max_output_tokens: int = 0,
    tokenizer_type: Literal["huggingface", "tiktoken"] | None = None,
    truncate_type: Literal["middle", "last_n_turns"] = "middle",
    buffer_tokens: int = 10,
    *,
    token_counter: TokenCounter | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    add_generation_prompt: bool = True,
) -> str | Conversation:
    token_counter = _build_compat_token_counter(
        token_counter=token_counter,
        tokenizer=tokenizer,
        tokenizer_type=tokenizer_type,
    )

    if truncate_type == "middle":
        if not isinstance(text, str):
            raise ValueError("If truncate type == middle, text must be a string")
        return _truncate_middle(
            text=text,
            token_counter=token_counter,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            buffer_tokens=buffer_tokens,
        )
    elif truncate_type == "last_n_turns":
        if not isinstance(text, Conversation):
            raise ValueError(
                "If truncate type == last_n_turns, text must be a Conversation"
            )
        return _truncate_last_n_turns(
            conversation=text,
            token_counter=token_counter,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            buffer_tokens=buffer_tokens,
        )
    else:
        raise NotImplementedError("Supported truncate_type: middle only")
