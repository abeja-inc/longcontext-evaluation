#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export BASE_URL="${BASE_URL:-http://${HOST}:${PORT}/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME}"
export MODEL_PATH="${MODEL_PATH}"

echo "[runner] starting vLLM server..."
vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --served-model-name "${MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size 2 \
  --dtype float32 \
  --reasoning-parser qwen3 \
  &
VLLM_PID=$!

cleanup() {
  echo "[runner] stopping vLLM server (pid=${VLLM_PID})..."
  kill -TERM "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ★ ここが重要: readiness 用の Authorization ヘッダ
AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")
fi

echo "[runner] waiting for vLLM to be ready..."
for i in $(seq 1 180); do
  if curl -sf "${AUTH_HEADER[@]}" "http://${HOST}:${PORT}/v1/models" > /dev/null; then
    echo "[runner] vLLM is ready."
    break
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[runner] vLLM server exited unexpectedly."
    exit 1
  fi

  sleep 1
done

curl -sf "${AUTH_HEADER[@]}" "http://${HOST}:${PORT}/v1/models" > /dev/null \
  || { echo "[runner] vLLM did not become ready"; exit 1; }

echo "[runner] running smoke test..."
python3 test_vllm_openai_api_compatible.py
echo "[runner] smoke test finished."
