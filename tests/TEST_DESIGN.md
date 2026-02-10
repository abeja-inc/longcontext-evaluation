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

#### `test_filter_long_inputs.py`: test for `_filter_long_inputs`
- `test_filter_long_inputs_keeps_order_and_returns_original_skip_indices`: `_filter_long_inputs(...)` が長さ超過と非超過の混在入力を受け取ったときに、非超過入力の元順序を保持し、スキップした要素の元インデックスを返し、件数の整合性 (`len(filtered_inputs) + len(skip_idx) == len(inputs)`) を満たすことを検証する。

#### test for `chat` and `completion`
- `test_chat_raises_runtime_error_when__chat_returns_too_few_responses`:
  verifies `BaseGenerator.chat` raises `RuntimeError` when `_chat` returns fewer responses than input conversations.
- `test_completion_raises_runtime_error_when__completion_returns_too_few_responses`:
  verifies `BaseGenerator.completion` raises `RuntimeError` when `_completion` returns fewer responses than input prompts.
- `test_chat_raises_runtime_error_when__chat_returns_too_many_responses`:
  verifies `BaseGenerator.chat` raises `RuntimeError` when `_chat` returns more responses than input conversations.
- `test_completion_raises_runtime_error_when__completion_returns_too_many_responses`:
  verifies `BaseGenerator.completion` raises `RuntimeError` when `_completion` returns more responses than input prompts.

### `test_kwargs_forwarding_to_count_tokens.py`: test for keyword arguments forwarding to `_count_tokens`
- Verify that `BaseGenerator._is_over_context_length` forwards keyword arguments to `_count_tokens`.
- Verify that `BaseGenerator._filter_long_inputs` forwards keyword arguments to `_count_tokens` for each input.
- Verify that `_count_tokens` is called once per input when filtering multiple inputs.

### `tests/llm_inference/test_token_counter.py`: test for `TokenCounter`
- `test_init_huggingface_uses_from_pretrained_with_defaults`
  - 目的: `tokenizer_type="huggingface"` で初期化した際に `AutoTokenizer.from_pretrained(...)` が `trust_remote_code=True` 付きで呼ばれ、`tokenizer_kwargs` 未指定時に空辞書が採用されることを確認する。
- `test_init_huggingface_forwards_custom_tokenizer_kwargs`
  - 目的: `tokenizer_kwargs` を指定した際に `AutoTokenizer.from_pretrained(...)` へ正しく透過的に渡されることを確認する。
- `test_init_tiktoken_prefers_encoding_for_model`
  - 目的: `tokenizer_type="tiktoken"` の初期化で `encoding_for_model(...)` が成功した場合、`get_encoding(...)` にフォールバックしないことを確認する。
- `test_init_tiktoken_falls_back_to_get_encoding_on_key_error`
  - 目的: `encoding_for_model(...)` が `KeyError` を送出した場合に `get_encoding(...)` へフォールバックする分岐を確認する。
- `test_init_raises_for_unsupported_tokenizer_type`
  - 目的: 未対応 `tokenizer_type` 指定時に `ValueError` が送出される異常系を確認する。
- `test_count_tokens_huggingface_conversation_uses_apply_chat_template`
  - 目的: HuggingFace 分岐の `Conversation` 入力で `apply_chat_template(tokenize=True, **kwargs)` が使われ、キーワード引数が透過されることを確認する。
- `test_count_tokens_huggingface_prompt_and_string_use_encode_without_special_tokens`
  - 目的: HuggingFace 分岐の `Prompt` / `str` 入力で `encode(..., add_special_tokens=False)` が使われることを確認する。
- `test_count_tokens_tiktoken_prompt_and_string_use_encode`
  - 目的: tiktoken 分岐の `Prompt` / `str` 入力で `encode(...)` が使われることを確認する。
- `test_count_tokens_tiktoken_conversation_counts_overhead_and_special_fields`
  - 目的: tiktoken 分岐の `Conversation` 入力で、メッセージ固定オーバーヘッド・`name` キー加算・`None` スキップ・非文字列 `str(...)` 化・末尾固定トークン加算の各分岐を網羅的に検証する。
