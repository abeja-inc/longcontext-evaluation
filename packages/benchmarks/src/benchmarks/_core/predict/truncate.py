import warnings
from typing import Any, Literal, overload

from llm_inference.data import Conversation


def count_conversation_tokens(
    conv: "Conversation",
    tokenizer: Any,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    chat_template_kwargs: dict[str, Any] = {},
) -> int:
    if tokenizer_type == "huggingface":
        token_ids = tokenizer.apply_chat_template(
            [m.model_dump() for m in conv.messages],
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        return len(token_ids)

    if tokenizer_type == "tiktoken":
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for m in conv.prompt:
            num_tokens += tokens_per_message
            for key, value in m.items():
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                num_tokens += len(tokenizer.encode(value, disallowed_special=()))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    raise NotImplementedError("Supported tokenizer_type: huggingface or tiktoken")


def _truncate_middle(
    text: str,
    tokenizer: Any,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    max_context_length: int,
    max_output_tokens: int,
    buffer_tokens: int = 10,
) -> str:
    if tokenizer_type == "huggingface":
        input_ids = tokenizer.encode(text)
    elif tokenizer_type == "tiktoken":
        input_ids = tokenizer.encode(text, disallowed_special=())
    else:
        raise NotImplementedError("Supported tokenizer_type: huggingface or tiktoken")
    input_length = len(input_ids)

    if input_length + max_output_tokens > max_context_length:
        max_len = max_context_length - max_output_tokens - buffer_tokens
        truncated_input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        if tokenizer_type == "huggingface":
            return tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
        elif tokenizer_type == "tiktoken":
            return tokenizer.decode(truncated_input_ids)
        else:
            raise NotImplementedError(
                "Supported tokenizer_type: huggingface or tiktoken"
            )
    else:
        return text


def _truncate_last_n_turns(
    conversation: Conversation,
    tokenizer: Any,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    max_context_length: int,
    max_output_tokens: int,
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

    # user のメッセージが一つだけ
    if first_user == last_user:
        if count_conversation_tokens(conversation, tokenizer, tokenizer_type) > budget:
            warnings.warn(
                "Even the single-user conversation exceeds the context budget"
            )
        return conversation

    # 必ず残すパート
    prefix_indices: list[int] = list(range(0, first_user))
    second_user: int = user_idxs[1]
    first_segment_indices: list[int] = list(range(first_user, second_user))
    last_user_indices: list[int] = [last_user]

    # 削る対象
    middle_segments: list[list[int]] = []
    for k in range(1, len(user_idxs) - 1):
        start = user_idxs[k]
        end = user_idxs[k + 1]  # next user
        middle_segments.append(list(range(start, end)))

    # Truncation process
    kept_indices: list[int] = []
    kept_indices.extend(prefix_indices)
    kept_indices.extend(first_segment_indices)
    kept_indices.extend(last_user_indices)
    kept_indices = sorted(set(kept_indices))

    mandatory_conv = Conversation(
        messages=[messages[i] for i in kept_indices],
        metadata=conversation.metadata,
    )
    if count_conversation_tokens(mandatory_conv, tokenizer, tokenizer_type) > budget:
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
            count_conversation_tokens(truncated_conversation, tokenizer, tokenizer_type)
            <= budget
        ):
            return truncated_conversation

        if kept_middle_count == 0:
            warnings.warn("Cannot truncate to fit the context budget")

        kept_middle_count -= 1


@overload
def truncate_text(
    text: str,
    tokenizer: Any,
    max_context_length: int,
    max_output_tokens: int,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    truncate_type: Literal["middle", "last_n_turns"] = "middle",
    buffer_tokens: int = 10,
) -> str: ...


@overload
def truncate_text(
    text: Conversation,
    tokenizer: Any,
    max_context_length: int,
    max_output_tokens: int,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    truncate_type: Literal["middle", "last_n_turns"] = "last_n_turns",
    buffer_tokens: int = 10,
) -> Conversation: ...


def truncate_text(
    text: str | Conversation,
    tokenizer: Any,
    max_context_length: int,
    max_output_tokens: int,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    truncate_type: Literal["middle", "last_n_turns"] = "middle",
    buffer_tokens: int = 10,
) -> str | Conversation:
    if truncate_type == "middle":
        if not isinstance(text, str):
            raise ValueError("If truncate type == middle, text must be a string")
        return _truncate_middle(
            text=text,
            tokenizer=tokenizer,
            tokenizer_type=tokenizer_type,
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
            tokenizer=tokenizer,
            tokenizer_type=tokenizer_type,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            buffer_tokens=buffer_tokens,
        )
    else:
        raise NotImplementedError("Supported truncate_type: middle only")
