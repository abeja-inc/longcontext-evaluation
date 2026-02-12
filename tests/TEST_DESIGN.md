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
- 方針: clientレスポンスの妥当性は仮定し、Generator層の入出力対応（入力順序と `Response.outputs[0].content` の位置対応）が維持されることのみを検証する。
- `test_chat_keeps_input_output_correspondence_for_normal_case`
  - 目的: `_chat` に長さ 3 の複数 `Conversation` を渡した通常系で、クライアントの応答内容 (`response:<input>`) が入力順序と 1:1 で対応して返却されることを確認する。
- `test_completion_keeps_input_output_correspondence_for_normal_case`
  - 目的: `completion` に長さ 3 の複数 `Prompt` を渡した通常系で、クライアントの応答内容 (`response:<input>`) が入力順序と 1:1 で対応して返却されることを確認する。
- `test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `_chat` の異常/境界系として中間入力のみ長文扱いにした場合、該当位置だけエラーメッセージに置換され、前後入力との対応関係・順序が崩れないことを確認する。
- `test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `completion` の異常/境界系として中間入力のみ長文扱いにした場合、該当位置だけエラーメッセージに置換され、前後入力との対応関係・順序が崩れないことを確認する。

### `tests/llm_inference/openai_api_compatible/test_openai_compatible_generator.py`: test for `OpenAICompatibleGenerator`
- 方針: clientレスポンスの妥当性は仮定し、Generator層の入出力対応（入力順序と `Response.outputs[0].content` の位置対応）が維持されることのみを検証する。
- `test_chat_keeps_input_output_correspondence_for_normal_case`
  - 目的: `_chat` に長さ 3 の複数 `Conversation` を渡した通常系で、クライアントの応答内容 (`response:<input>`) が入力順序と 1:1 で対応して返却されることを確認する。
- `test_completion_keeps_input_output_correspondence_for_normal_case`
  - 目的: `completion` に長さ 3 の複数 `Prompt` を渡した通常系で、クライアントの応答内容 (`response:<input>`) が入力順序と 1:1 で対応して返却されることを確認する。
- `test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `_chat` の異常/境界系として中間入力のみ長文扱いにした場合、該当位置だけエラーメッセージに置換され、前後入力との対応関係・順序が崩れないことを確認する。
- `test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `completion` の異常/境界系として中間入力のみ長文扱いにした場合、該当位置だけエラーメッセージに置換され、前後入力との対応関係・順序が崩れないことを確認する。


### `tests/llm_inference/vllm_offline_inference/test_vllm_offline_generator.py`: test for `VLLMOfflineGenerator`
- 方針: Generator層の対応関係保証を検証し、backend品質は仮定する。`vllm.LLM` は fake 実装に差し替え、deterministic な `response:<input>` を返すことで入出力順序とスキップ位置の整合性にのみ注目する。
- `test_chat_keeps_input_output_correspondence_for_normal_case`
  - 目的: `_chat` に長さ 3 の複数 `Conversation` を渡した通常系で、入力順序と `Response.outputs[0].content` の 1:1 対応が維持されることを確認する。
- `test_completion_keeps_input_output_correspondence_for_normal_case`
  - 目的: `completion` に長さ 3 の複数 `Prompt` を渡した通常系で、入力順序と `Response.outputs[0].content` の 1:1 対応が維持されることを確認する。
- `test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `_chat` で middle input のみ長文扱いにした混在系 (`[ok, too_long, ok]`) で、該当位置に `default_too_long_input_error_message` が入り、前後要素との index 対応が崩れないことを確認する。
- `test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `completion` で middle input のみ長文扱いにした混在系 (`[ok, too_long, ok]`) で、該当位置に `default_too_long_input_error_message` が入り、前後要素との index 対応が崩れないことを確認する。

### `tests/llm_inference/sglang_offline_inference/test_sglang_offline_generator.py`: test for `SGLangOfflineGenerator`
- 方針: backend品質は仮定し、Generator層の対応保証（入力順序・スキップ位置・`max_tokens` 反映）に限定して検証する。`ServerArgs`/`Engine`/tokenizer を fake 化し、`Engine.generate` は deterministic な `response:<input>` を返す。
- `test_chat_keeps_input_output_correspondence_for_normal_case`
  - 目的: `_chat` 通常系で入力3件と出力3件の index 対応が崩れず、chat 文字列整形経由で `generate` が呼ばれることを確認する。
- `test_completion_keeps_input_output_correspondence_for_normal_case`
  - 目的: `_completion` 通常系で入力3件と出力3件の index 対応が崩れず、prompt 配列経由で `generate` が呼ばれることを確認する。
- `test_chat_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `_chat` の混在系 (`[ok, too_long, ok]`) で、too long 位置のみ `default_too_long_input_error_message` となり、前後要素のずれや欠落がないことを確認する。
- `test_completion_keeps_input_output_correspondence_when_middle_input_is_too_long`
  - 目的: `_completion` の混在系 (`[ok, too_long, ok]`) で、too long 位置のみ `default_too_long_input_error_message` となり、前後要素のずれや欠落がないことを確認する。
- `test_generate_raises_value_error_when_sampling_param_n_is_greater_than_one`
  - 目的: `_generate` 異常系として `sampling_params={"n": 2}` を渡したときに `ValueError` が送出されることを確認する（`n>1` 非対応制約の防御）。

## `tests/benchmarks/ruler/synthesize_dataset/niah/test_niah_haystack_selection.py`: test for NIAH haystack composition
- `test_gen_one_sample_noise_haystack_keeps_noise_as_main_component`
  - 目的: `type_haystack="noise"` で `_gen_one_sample(...)` を固定 seed で呼び、`content.user_prompt` 本文が「ノイズ文の繰り返し + 挿入 needle」で構成されることを確認する。
  - 観点: 抽出した needle 候補が `extra_fields['needles']` と一致し、ノイズ文の出現数が `num_units` を維持することを検証する。
- `test_gen_one_sample_needle_haystack_keeps_targets_identifiable_as_inserted`
  - 目的: `type_haystack="needle"` で `_gen_one_sample(...)` を固定 seed で呼び、背景 needle と target needle が混在する中でも target needle 群が本文内で識別可能であることを確認する。
  - 観点: `extra_fields['needles']` が本文から抽出した needle 候補集合に欠落なく含まれ、かつ候補全体と完全同一化しない（背景 needle が別途存在する）ことを検証する。

## `tests/benchmarks/ruler/synthesize_dataset/qa/test_qa_context_selection.py`: test for QA context composition
- `test_gen_one_sample_qa_keeps_gold_context_in_prompt_and_target_context`
  - 目的: QA の `_gen_one_sample(...)` を固定 seed で実行し、gold context が `content.user_prompt` の本文（Document 群）に確実に含まれ、`extra_fields['target_context']` と整合することを確認する。
- `test_gen_one_sample_qa_mixes_gold_and_distractors_when_available`
  - 目的: distractor が利用可能な条件で `_gen_one_sample(...)` を固定 seed で実行し、生成本文が gold のみで単一化されず、gold と distractor が混在することを確認する。
