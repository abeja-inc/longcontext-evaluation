# longcontext-evaluation
## Overview
LLM のロングコンテキスト処理性能を評価するためのベンチマーク実装

## Features
以下の評価をサポート
- RULER の NIAH（英） および QA（日英） タスク
- LongBench v2（英）
- OpenAI-MRCR（日英）
- experimental
    - Nemotron-Persona_Japanese_QA
    - Context-Poisoning-Make-10-Puzzle

## Installation
### For Docker
```sh
# Check import
docker compose -f containers/docker/docker-compose.yaml run --rm vllm

# Start container
docker compose -f containers/docker/docker-compose.yaml run --rm --entrypoint /bin/bash vllm
```

### For slurm
```sh
sbatch containers/enroot/sbatch_make_image.sh
```

## Download models
### HuggingFace
```
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
hf download openai/gpt-oss-20b --local-dir models/gpt-oss-20b
```

## Dataset Preparation
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

### vLLM Offline Inference
```sh
python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/vllm_offline.yml
```

### OpenAI API
```sh
export OPENAI_API_KEY=your_key
python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/openai_api.yml
```

### vLLM OpenAI-compatible Server
```sh
export API_KEY=your_key
export BASE_URL=http://localhost:8000/v1
python3 scripts/benchmarks/run.py \
    --config scripts/benchmarks/run_configs/vllm_openai_compatible.yml
```
