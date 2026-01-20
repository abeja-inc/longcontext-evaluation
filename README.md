# longcontext-evaluation
## Overview
LLM のロングコンテキスト処理性能を評価するためのベンチマーク実装

## Features
以下の評価をサポート
- RULER の NIAH（英） および QA（日英） タスク
- LongBench v2（英）
- OpenAI-MRCR（日英）

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
