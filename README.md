# longcontext-evaluation
各種ベンチマーク・タスクの選定理由等は以下のブログに記載しています。

[GENIAC3期のLLM開発で使用したロングコンテキスト評価のベンチマーク公開](https://tech-blog.abeja.asia/entry/geniac3-longcontext-benchmark-pr-202603)

## Overview
LLM のロングコンテキスト処理性能を評価するためのベンチマーク実装

設計の概要は [DESIGN.md](DESIGN.md) を参照してください。

## Features
以下の評価をサポート
- RULER: [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
  - NIAH（英） および QA（日英） タスク
    - QA datasets: SQuAD, HotpotQA, JSQuAD, JEMHopQA
- LongBench v2: [THUDM/LongBench](https://github.com/THUDM/LongBench)
  - 英
- OpenAI-MRCR: [openai/mrcr](https://huggingface.co/datasets/openai/mrcr)
  - 日英（日本語は英語データセットを翻訳）: [abejaOpenAI-MRCR-Translation-JPN](https://huggingface.co/datasets/abeja/OpenAI-MRCR-Translation-JPN)
- experimental
    - Nemotron-Persona_Japanese_QA (To be added)
      - 日
    - Context-Poisoning-Make-10-Puzzle (To be added)
      - 日

## 補助機能
- `inference_mode`: chat mode (text generation using chat api), completion-mode (text generation using completion api)
    - データセットの形式に依存するためベンチマークごとにサポート状況が異なる
        - chat: LongBench v2, OpenAI-MRCR, RULER
        - completion: RULER
- `use_truncate`: Truncation の ON/OFF を制御
  - `use_truncate: false` に設定すると、 Truncation なしでの性能評価が可能
  -  `truncate_buffer_tokens`: Truncation の際のトークン数カウントにおいて、指定する値だけバッファを設けて Truncation する
    - `truncate_buffer_tokens: 10` (Default) とすると、計測されたトークン数 + 10トークンが入力プロンプトのトークン数と見做される（トークン数のカウントが正確にできない時に使用する）
- `truncate_type`: 入力プロンプト＋最大出力トークン数がモデルのコンテキスト長を超える場合に、入力プロンプトの一部を切り取る
    - `middle` truncation (for LongBench v2, RULER): 入力プロンプトの中央を切り取る。プロンプトがテキスト形式のデータに使用。
    - `last_n_turns` truncation (for MRCR): 入力プロンプトの末尾の会話（最後のユーザプロンプトを除く）からNターンを切り取る。プロンプトが Messages (`[{"role": "user", "content": "..."}, ...]`)形式のデータに使用。
- require_reasoning: Reasoning モデルの評価において、 reasoning content がパースできない場合に不正解とする
    - `require_reasoning: true` を指定すると、`output_reasoning` が欠損したサンプルは全ベンチマーク共通で不正解（score=0.0）として扱います。
- filtering long inputs:　入力プロンプトがモデルのコンテキスト長を超える場合に、事前にフィルタリングする機能
  - LLM 側の推論設定にある `buffer_tokens: 0` はフィルタリング時のトークン数のカウントにバッファを設けるために用いる

## Installation
### For Docker
```sh
# Check import
docker compose -f containers/docker/docker-compose.yaml run --rm vllm

# Start container
docker compose -f containers/docker/docker-compose.yaml run --name lc_eval --rm --entrypoint /bin/bash vllm
```

#### Optional) Update dependencies
依存関係を更新したい場合、

```sh
docker compose -f containers/docker/docker-compose-update-dependancies.yaml run --rm --entrypoint /bin/bash vllm
```
でベースイメージのコンテナを作成し、

1. `uv add ...` / `uv remove ...`
2. `uv pip compile pyproject.toml --constraint base-image-constraints.txt`
3. `uv lock`

を行って `uv.lock` を更新してください。その上で、ベンチマーク実行用のコンテナを image の build から作成し直してください。

### For slurm
TBD

## Download models
### HuggingFace
```
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
hf download openai/gpt-oss-20b --local-dir models/gpt-oss-20b
```
ローカル実行や smoke test では `/workspace/models/` を想定するテストを用意しているため、デフォルトの設定ではローカルの `models/` を `/workspace/models` （コンテナ内）でマウントする形になっています。

## Dataset Preparation
ダウンロードしたデータセットはデフォルトで `datasets/benchmarks` に配置されます（`scripts/benchmarks/run_configs/base.yaml` の `dataset_root` 参照）。

### LongBench v2
評価用データセットをダウンロード
```sh
python3 scripts/benchmarks/LongBench_v2/download_evaluation_dataset/download_dataset.py \
    --config scripts/benchmarks/LongBench_v2/download_evaluation_dataset/config.yaml
```

### OpenAI-MRCR
評価用データセットをダウンロード
```sh
python3 scripts/benchmarks/OpenAI_MRCR/download_evaluation_dataset/download_dataset.py \
    --config scripts/benchmarks/OpenAI_MRCR/download_evaluation_dataset/config.yaml
```


### RULER
評価用データセットをダウンロード
```sh
python3 scripts/benchmarks/RULER/download_evaluation_dataset/download_dataset.py \
    --config scripts/benchmarks/RULER/download_evaluation_dataset/config.yaml
```

ダウンロードしたデータセットをもとにロングコンテキスト評価データを合成
```sh
## Needle in a haystack
python3 scripts/benchmarks/RULER/synthesize_evaluation_dataset/niah/make_dataset.py \
    --config scripts/benchmarks/RULER/synthesize_evaluation_dataset/niah/config.yaml
    
## QA
python3 scripts/benchmarks/RULER/synthesize_evaluation_dataset/qa/make_dataset.py \
    --config scripts/benchmarks/RULER/synthesize_evaluation_dataset/qa/config.yaml
```

## Benchmark Execution
統一実行スクリプトでベンチマーク評価を実行します。

**Configuration**
- benchmark settings: [base_config.yaml](scripts/benchmarks/run_configs/base.yaml)
- model setting examples:
    - [OpenAI API](scripts/benchmarks/run_configs/openai_api.yml)
    - [vLLM OpenAI-compatible Server](scripts/benchmarks/run_configs/openai_compatible.yml)
    - [vLLM Offline Inference](scripts/benchmarks/run_configs/vllm_offline.yml)

**Execution**
- [run.py](scripts/benchmarks/run.py)

**Outputs**
- 実行結果は `scripts/benchmarks/run_configs/base.yaml` の `output_root` 配下に `run_name` で保存されます。
- `run_name` は各 config で定義するか、`--config` 内の環境変数で上書きできます。

### vLLM Offline Inference
```sh
CUDA_VISIBLE_DEVICES=8 python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/vllm_offline.yaml
```

### OpenAI API
```sh
export OPENAI_API_KEY=your_key
export OPENAI_MODEL_NAME=gpt-4o-2024-11-20
python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/openai_api.yaml
```

### vLLM OpenAI-compatible Server
```sh
export API_KEY=your_key
export BASE_URL=http://localhost:8000/v1
python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/vllm_openai_compatible.yaml
```

## Repository layout (high-level)
- `packages/benchmarks`: ベンチマーク評価ロジック
- `packages/llm_inference`: モデル推論の抽象化レイヤ
- `scripts/benchmarks`: データセットのダウンロード/合成/評価実行
- `smoke_tests/llm_inference`: 推論モジュールの簡易動作チェック

## NOTICE
このプロジェクトは以下の既存ベンチマークの実装をリファクタリング・拡張したものを一部含んでいます。

- [NVIDIA/RULER](https://github.com/NVIDIA/RULER), which is licensed under the Apache License 2.0.
- [THUDM/LongBench](https://github.com/THUDM/LongBench), which is licensed under the MIT License.

このリポジトリ自体のライセンスは [LICENSE](LICENSE) に記載の Apache License 2.0 です。
派生元コードに関する著作権表示・ライセンステキストは [NOTICE](NOTICE) と [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) に記載しています。
既存ベンチマーク実装のリファクタリング・拡張を含むファイルには、個々のファイル冒頭に派生元の著作権・ライセンス情報を記載しています。

- [NOTICE](NOTICE)
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
