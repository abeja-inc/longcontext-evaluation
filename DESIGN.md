# DESIGN.md
このドキュメントは、`packages/llm_inference` と `packages/benchmarks` の実装（設計思想・構成・使い方）を整理したものです。コードの読み取り結果に基づき、どこで責務を分けているか、どのように拡張しやすくしているかに焦点を当てています。

# llm_inference
## 設計意図
- **複数の推論バックエンドを統一インターフェースで扱う**ことを主目的に、`BaseGenerator` を中心とした設計にしています。オンライン（OpenAI API / OpenAI互換API）とオフライン（vLLM / SGLang）を同じ `chat` / `completion` で呼び出せるようにすることで、ベンチマーク側のロジックを単純化しています。([base.py](packages/llm_inference/src/llm_inference/base.py)、[openai_api.py](packages/llm_inference/src/llm_inference/openai_api.py)、[vllm_offline_inference.py](packages/llm_inference/src/llm_inference/vllm_offline_inference.py))
- **長すぎる入力の扱いを共通化**し、モデルの文脈長を超える入力をスキップ・警告・簡易的なエラーメッセージ返却で統一します。これにより、上位層（benchmarks）での例外処理を最小化しています。([base.py](packages/llm_inference/src/llm_inference/base.py))
- **推論結果の「思考過程（reasoning）」の分離**を可能にするため、`reasoning_parser` をオプションとして提供し、モデル固有の出力フォーマットに対応できるようにしています。([vllm_offline_inference.py](packages/llm_inference/src/llm_inference/vllm_offline_inference.py)、[reasoning_parser](packages/llm_inference/src/llm_inference/reasoning_parser))

## デザイン・構成
- **データモデル**: `Prompt` / `Conversation` / `Response` を Pydantic モデルで定義し、入力と出力の形を統一しています。ベンチマーク側もこれらの型に依存する設計です。([data.py](packages/llm_inference/src/llm_inference/data.py))
- **BaseGenerator**:
  - `_count_tokens`（抽象）でトークン数を計測し、`_is_over_context_length` / `_filter_long_inputs` で文脈長超過をチェック。([base.py](packages/llm_inference/src/llm_inference/base.py))
  - `_chat` / `_completion` を実装し、`chat` / `completion` の公開メソッドで出力件数を検証します。([base.py](packages/llm_inference/src/llm_inference/base.py))
- **具体的な Generator 実装**:
  - `OpenAIGenerator`: OpenAI Responses API を用いたオンライン推論。メッセージ形式とテキスト形式の両方で呼び出します。([openai_api.py](packages/llm_inference/src/llm_inference/openai_api.py))
  - `OpenAICompatibleGenerator`: OpenAI互換APIに対応しつつ、HuggingFace tokenizer を使う前提で `tokenizer_type` を切り替えます。([openai_api_compatible.py](packages/llm_inference/src/llm_inference/openai_api_compatible.py))
  - `VLLMOfflineGenerator`: vLLM の `LLM` を使ったオフライン推論。`chat`/`generate` を切り替え、必要なら reasoning parser で出力を分割します。([vllm_offline_inference.py](packages/llm_inference/src/llm_inference/vllm_offline_inference.py))
  - `SGLangOfflineGenerator`: SGLang の `Engine` を使ったオフライン推論。複数出力 `n>1` をサポートしない前提で制約を明示しています。([sglang_offline_inference.py](packages/llm_inference/src/llm_inference/sglang_offline_inference.py))
- **Factory**: `get_generator` でモジュール名から generator を動的に読み込み、`BaseGenerator` のサブクラスを一つだけ返します。新しい generator の追加が容易です。([__init__.py](packages/llm_inference/src/llm_inference/__init__.py))
- **Reasoning parser**: `qwen3` や `openai_gptoss` のように、出力の区切り文字を基に `reasoning_content` と `content` を分割します。([reasoning_parser](packages/llm_inference/src/llm_inference/reasoning_parser))

## 基本的な使い方
- **Generator の生成**: `get_generator(type=...)` で `llm_inference` 側の実装を選択します。OpenAI API を利用する場合は `OpenAI` クライアントを渡します。([__init__.py](packages/llm_inference/src/llm_inference/__init__.py)、[scripts/benchmarks/run.py](scripts/benchmarks/run.py))
- **推論呼び出し**:
  - チャット形式: `generator.chat(conversations=[Conversation(...), ...])`
  - 完了形式: `generator.completion(prompts=[Prompt(...), ...])`
  - いずれも `Response` が返り、`outputs[0].content` に生成結果が入ります。([base.py](packages/llm_inference/src/llm_inference/base.py)、[data.py](packages/llm_inference/src/llm_inference/data.py))

# benchmarks
## 設計意図
- **ベンチマークごとの処理差分を最小化**するため、共通の `BaseBenchmarkRunner` を中心に「予測 → 評価 → 集計 → 保存」を一貫して処理します。各ベンチマークは `_run_subtask` と `_evaluate_subtask` の実装に集中できます。([_core/runner.py](packages/benchmarks/src/benchmarks/_core/runner.py))
- **設定駆動**でデータセットや出力先を切り替えられるよう、`BenchmarkConfig` / `TaskConfig` / `SubtaskConfig` による構造化設定を採用しています。([config.py](packages/benchmarks/src/benchmarks/config.py))
- **文脈長超過時の取り扱いを統一**するため、共通の `truncate_text` を提供し、ベンチマーク固有の設定（`BaseSettings`）から制御します。([_core/predict/truncate.py](packages/benchmarks/src/benchmarks/_core/predict/truncate.py)、[_core/settings.py](packages/benchmarks/src/benchmarks/_core/settings.py))

## デザイン・構成
- **エントリポイント**:
  - `run_benchmarks` が複数ベンチマークを順次実行し、`runner_factory` から該当 runner を取得します。([run.py](packages/benchmarks/src/benchmarks/run.py)、[runner_factory.py](packages/benchmarks/src/benchmarks/runner_factory.py))
- **共通コア（_core）**:
  - `BaseBenchmarkRunner`: 予測・評価・集計・保存の共通フロー。Pydantic でサブタスク設定の妥当性も検証します。([_core/runner.py](packages/benchmarks/src/benchmarks/_core/runner.py))
  - `mean_score_by_group` / `mean_score_by_group_and_context_bin`: 結果の集計と文脈長 bin 化。([_core/evaluate/mean_score.py](packages/benchmarks/src/benchmarks/_core/evaluate/mean_score.py))
  - `OutputsTable` / `MeanScoreByLengthTable` などのテーブル定義と `BenchmarkResults` によるまとめ。([_core/evaluate/table.py](packages/benchmarks/src/benchmarks/_core/evaluate/table.py))
  - `save_to_local` / `push_to_wandb`: JSONL/CSV で保存するローカル出力と W&B への送信。([_core/save_table/to_local.py](packages/benchmarks/src/benchmarks/_core/save_table/to_local.py)、[_core/save_table/to_wandb.py](packages/benchmarks/src/benchmarks/_core/save_table/to_wandb.py))
- **ベンチマーク実装**:
  - `longbench_v2` / `mrcr` / `ruler` それぞれが `Runner` と `Settings` を持ちます。`_run_subtask` でデータ読み込みと推論、`_evaluate_subtask` でスコアリングを行います。([longbench_v2/runner.py](packages/benchmarks/src/benchmarks/longbench_v2/runner.py)、[mrcr/runner.py](packages/benchmarks/src/benchmarks/mrcr/runner.py)、[ruler/runner.py](packages/benchmarks/src/benchmarks/ruler/runner.py))
  - `Settings` で truncation などの推奨／非推奨をバリデーションしています。([longbench_v2/settings.py](packages/benchmarks/src/benchmarks/longbench_v2/settings.py)、[mrcr/settings.py](packages/benchmarks/src/benchmarks/mrcr/settings.py)、[ruler/settings.py](packages/benchmarks/src/benchmarks/ruler/settings.py))

## 基本的な使い方
- **実行スクリプト**: `scripts/benchmarks/run.py` が実運用の入り口です。YAML の config を読み込み、`llm_inference.get_generator` で LLM 推論エンジンを生成し、`run_benchmarks` を実行します。([scripts/benchmarks/run.py](scripts/benchmarks/run.py))
- **設定の流れ**:
  1. YAML で `dataset_root` / `output_root` / `benchmarks` を定義。([scripts/benchmarks/run.py](scripts/benchmarks/run.py))
  2. `BenchmarkConfig` / `TaskConfig` / `SubtaskConfig` に変換し、実行時に runner へ渡す。([scripts/benchmarks/run.py](scripts/benchmarks/run.py)、[config.py](packages/benchmarks/src/benchmarks/config.py))
  3. 推論結果は `output_root/<run_name>/tables` 以下に JSONL として保存されます。([_core/runner.py](packages/benchmarks/src/benchmarks/_core/runner.py)、[_core/save_table/to_local.py](packages/benchmarks/src/benchmarks/_core/save_table/to_local.py))
