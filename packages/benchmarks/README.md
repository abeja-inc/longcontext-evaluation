# benchmarks

## Token counting policy

- Benchmarks must not reimplement token counting formulas.
- Truncation and context-budget checks must use `llm_inference.token_counter.TokenCounter`.
- Runner code should pass `generator.token_counter` into truncate utilities so benchmark and inference share identical counting behavior.

## Chat/completion kwargs behavior

- Chat truncation paths can pass `chat_template_kwargs` and `add_generation_prompt` to align with inference-time template behavior.
- Completion truncation (middle truncation) still uses the tokenizer behind the shared `TokenCounter`.
