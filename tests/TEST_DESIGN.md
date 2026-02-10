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

### `tests/llm_inference/openai_api/test_openai_generator.py`: test for `OpenAIGenerator`
- `test_count_tokens_prompt_uses_tokenizer_encode`
  - 目的: `Prompt` 入力で `_count_tokens` が tokenizer の `encode(prompt)` 長をそのまま返すことを確認する。
- `test_count_tokens_conversation_counts_overhead_special_fields_and_footer`
  - 目的: `Conversation` 入力で OpenAI chat 形式の固定オーバーヘッド (`tokens_per_message=3`)・`name` キー加算・`None` スキップ・非文字列の `str(...)` 化・末尾 `+3` を含む合計トークン数計算を検証する。
- `test_count_tokens_raises_type_error_for_unsupported_input`
  - 目的: サポート外入力型に対して `_count_tokens` が `TypeError` を送出する異常系を確認する。
- `test_call_response_api_calls_openai_and_builds_response_with_usage_and_input_metadata`
  - 目的: `_is_over_context_length=False` の通常経路で `client.responses.create` への引数 (`model`, `input`, `max_output_tokens`) が正しく渡され、`output_text` が `Response.outputs[0].content` に格納され、`input.metadata` と `usage.model_dump()` が `Response.metadata` 上で共存できることを確認する。
- `test_call_response_api_returns_too_long_error_without_calling_api`
  - 目的: `_is_over_context_length=True` の長文入力経路で API が呼ばれず、`default_too_long_input_error_message` を含む `Response` が返ることを検証する。
- `test_call_response_api_skips_usage_metadata_when_usage_is_none`
  - 目的: `api_response.usage is None` のとき `Response.metadata` に `usage` が追加されないことを確認する。
- `test_chat_forwards_inputs_and_long_input_filter_kwargs`
  - 目的: `_chat` が `inputs` と `long_input_filter_kwargs`、および追加 kwargs を `_call_response_api` に透過して委譲することを spy で検証する。
- `test_completion_forwards_inputs_and_long_input_filter_kwargs`
  - 目的: `_completion` が `inputs` と `long_input_filter_kwargs`、および追加 kwargs を `_call_response_api` に透過して委譲することを spy で検証する。
