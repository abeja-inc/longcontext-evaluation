from typing import Any, Literal


def _truncate_middle(
    text: str,
    tokenizer: Any,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    max_context_length: int,
    max_output_tokens: int,
    buffer_tokens: int = 10,
) -> str:
    input_ids = tokenizer.encode(text)
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


def truncate_text(
    text: str,
    tokenizer: Any,
    max_context_length: int,
    max_output_tokens: int,
    tokenizer_type: Literal["huggingface", "tiktoken"],
    truncate_type: Literal["middle"],
    buffer_tokens: int = 10,
) -> str:
    if truncate_type == "middle":
        return _truncate_middle(
            text=text,
            tokenizer=tokenizer,
            tokenizer_type=tokenizer_type,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            buffer_tokens=buffer_tokens,
        )
    else:
        raise NotImplementedError("Supported truncate_type: middle only")
