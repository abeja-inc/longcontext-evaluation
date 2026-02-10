# Test Design

## `tests/llm_inference`
### `/base`
#### test for `_is_over_context_length`
- `test_is_over_context_length_allows_equal_boundary`
  - 目的: `input_tokens + max_output_tokens + buffer_tokens == max_context_length` の境界条件で、`BaseGenerator._is_over_context_length(...)` が `False` を返すことを確認する。
  - 観点: `Prompt` と `Conversation` の両入力型で同一挙動であることを検証する。
- `test_is_over_context_length_returns_true_when_total_exceeds_by_one`
  - 目的: 上記合計が `max_context_length + 1` のとき、`BaseGenerator._is_over_context_length(...)` が `True` を返すことを確認する。
  - 観点: `Prompt` と `Conversation` の両入力型で同一挙動であることを検証する。

#### test for `_filter_long_inputs`
- `test_filter_long_inputs_keeps_order_and_returns_original_skip_indices`: `_filter_long_inputs(...)` が長さ超過と非超過の混在入力を受け取ったときに、非超過入力の元順序を保持し、スキップした要素の元インデックスを返し、件数の整合性 (`len(filtered_inputs) + len(skip_idx) == len(inputs)`) を満たすことを検証する。

#### test for `chat` and `completion`
- `test_chat_raises_runtime_error_when__chat_returns_too_few_responses`:
  verifies `BaseGenerator.chat` raises `RuntimeError` when `_chat` returns fewer responses than input conversations.
- `test_completion_raises_runtime_error_when__completion_returns_too_few_responses`:
  verifies `BaseGenerator.completion` raises `RuntimeError` when `_completion` returns fewer responses than input prompts.
