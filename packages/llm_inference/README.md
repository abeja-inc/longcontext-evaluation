# llm_inference

## Token counting design

- Token length checks are centralized in `llm_inference.token_counter.TokenCounter`.
- `BaseGenerator._is_over_context_length` delegates to `self.token_counter` so every generator uses the same counting path.
- `self.tokenizer` / `self.tokenizer_type` remain for backward compatibility, but context-length decisions must be made via `TokenCounter`.

## `TokenCounter` API

`TokenCounter` explicitly supports both input types:

- `Prompt` (completion-style text)
- `Conversation` (chat-style messages)

For chat counting, use:

- `chat_template_kwargs`: forwarded to Hugging Face `apply_chat_template`
- `add_generation_prompt`: explicitly included/excluded for consistent length checks

You can initialize `TokenCounter` with either:

- model/encoding name (`TokenCounter(...)`)
- an existing tokenizer object (`TokenCounter.from_tokenizer(...)`)
