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

## `tests/benchmarks/ruler/synthesize_dataset/qa/test_base_qa_generator_sample.py`: test for `BaseQADatasetGenerator._gen_one_sample`
- `test_gen_one_sample_includes_gold_docs_and_answer_candidates`
  - 目的: 固定データを返すテスト用 `BaseQADatasetGenerator` サブクラスを使い、生成サンプルの `target_context` に `context_indices` の全文書が含まれること、`content.outputs` が正解候補と一致すること、`target_depth_percent` が `-1.0` または `0..100` に収まることを検証する。
- `test_gen_one_sample_boundary_cases_for_document_count_and_prompt_order`
  - 目的: `num_units` が gold 文書数より小さい場合と全 context 数より大きい場合の境界ケースで、`document_prompt` の連番 (`Document 1..N`) と文書ブロック順序が整合することを検証する。

## `tests/benchmarks/test_reasoning_requirement.py`: test for reasoning parser requirement in benchmark metrics
- `test_eval_returns_zero_when_reasoning_required_and_missing`
  - 目的: `settings.require_reasoning=True` かつ `output_reasoning is None` のとき、メトリクス本体を呼ばずに `BaseMetrics.eval` が 0.0 を返すことを確認する。
- `test_eval_runs_metric_when_reasoning_required_and_present`
  - 目的: `settings.require_reasoning=True` で `output_reasoning` が存在する場合は通常どおり各メトリクス関数へ処理が委譲されることを確認する。


## `tests/benchmarks/test_runner_require_reasoning_setting.py`: test for settings-driven reasoning behavior without runner kwarg forwarding
- `test_run_does_not_forward_require_reasoning_kwarg`
  - 目的: `BaseBenchmarkRunner.run` が `_evaluate_subtask` に `require_reasoning` kwargs を渡さないことを確認する。
- `test_run_scoring_still_uses_settings_require_reasoning_false`
  - 目的: `settings.require_reasoning=False` のとき、`output_reasoning` 欠損でもメトリクス実行結果（1.0）が返ることを確認する。
- `test_run_scoring_still_uses_settings_require_reasoning_true`
  - 目的: `settings.require_reasoning=True` のとき、`output_reasoning` 欠損が共通ガードで 0.0 扱いになることを確認する。

- `test_eval_uses_settings_field_even_if_kwargs_disagree`
  - 目的: `require_reasoning` 引数が渡されても評価判定は `settings.require_reasoning` を優先して行われることを確認する。

## `tests/benchmarks/longbench_v2/evaluate/test_metrics.py`: test for `LongBenchV2Metrics`
- `test_parsed_answer_match_valid_extraction_with_parenthesized_label`
  - 目的: `"The correct answer is (A)"` 形式が正しく抽出され、gold `A` と一致して 1.0 になることを確認する。
- `test_parsed_answer_match_valid_extraction_with_plain_label`
  - 目的: `"The correct answer is B"` 形式が正しく抽出され、gold `B` と一致して 1.0 になることを確認する。
- `test_parsed_answer_match_accepts_decorated_output_with_asterisks`
  - 目的: 出力が `*...*` で装飾されていても `*` を除去したうえで抽出され、正解判定できることを確認する。
- `test_parsed_answer_match_empty_output_returns_policy_values`
  - 目的: `output.output` が空文字のとき、`compensate_missing=False` で 0.0、`True` で 0.25 を返す欠損補償ポリシーを確認する。
- `test_parsed_answer_match_default_error_message_returns_policy_values`
  - 目的: `output.output` が `default_error_message` と一致する場合も空文字と同じ欠損補償ポリシーになることを確認する。
- `test_parsed_answer_match_parse_failure_returns_policy_values`
  - 目的: 回答形式にパース失敗した場合、`compensate_missing` の値に応じて 0.0/0.25 を返すことを確認する。
- `test_parsed_answer_match_invalid_gold_answer_returns_zero`
  - 目的: gold が `A-D` 以外の不正値のとき、常に 0.0 を返すことを確認する。
- `test_parsed_answer_match_label_mismatch_returns_zero`
  - 目的: パース結果と gold ラベルが不一致の場合に 0.0 を返すことを確認する。
- `test_eval_parsed_answer_match_uses_settings_metric_kwargs`
  - 目的: 公開メソッド `eval_parsed_answer_match(...)` が `settings.metric_kwargs` を内部評価に透過し、`compensate_missing=True` が有効になることを確認する。
