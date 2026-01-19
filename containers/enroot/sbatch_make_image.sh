#!/usr/bin/env bash
#SBATCH -J make_enroot_image
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err

set -euo pipefail

TAG="${TAG:-latest}"
SERVICE_NAME="${SERVICE_NAME:-vllm}"

# image name must match docker-compose.yaml's `image:`
IMAGE_REPO="${IMAGE_REPO:-longcontext-evaluation-vllm-openai}"
IMAGE_REF="${IMAGE_REPO}:${TAG}"

# Where to place sqsh
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "[ERROR] Not in a git repository. Run from inside the repo." >&2
  exit 1
fi

DEFAULT_OUT_DIR="${REPO_ROOT}/images"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"

# Compose file path (relative to repo root)
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/containers/docker/docker-compose.yaml}"

# Optional: name for enroot container (only used if you choose to enroot create)
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d-%H%M%S')}"
ENROOT_NAME="${ENROOT_NAME:-${IMAGE_REPO//[:\/]/_}_${TAG}}"

# Packages to import-check (edit as you like)
IMPORT_CHECK_MODULES=(
  "benchmarks"
  "llm_inference"
)

usage() {
  cat <<EOF
Usage:
  sbatch containers/enroot/sbatch_make_image.sh

Env overrides:
  TAG=latest|dev|...                  Tag for the docker image and sqsh
  IMAGE_REPO=longcontext-evaluation-vllm-openai
  SERVICE_NAME=vllm                   Compose service name (default: vllm)
  COMPOSE_FILE=/path/to/docker-compose.yaml
  OUT_DIR=/path/to/output/dir
  ENROOT_NAME=name_for_enroot_create

Notes:
  - This script builds docker image via docker compose, then imports it to enroot (.sqsh),
    then runs an import check inside enroot.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${OUT_DIR}"

echo "==============================================="
echo "[INFO] Repo root     : ${REPO_ROOT}"
echo "[INFO] Compose file  : ${COMPOSE_FILE}"
echo "[INFO] Service       : ${SERVICE_NAME}"
echo "[INFO] Docker image  : ${IMAGE_REF}"
echo "[INFO] Output dir    : ${OUT_DIR}"
echo "==============================================="

# Build docker image
cd "${REPO_ROOT}"

if command -v docker >/dev/null 2>&1; then
  echo "[INFO] Using docker"
  TAG="${TAG}" docker compose -f "${COMPOSE_FILE}" build "${SERVICE_NAME}"
elif command -v podman >/dev/null 2>&1; then
  echo "[INFO] docker not found; using podman (compose requires podman-compose or docker-compose equivalent)"
  echo "[ERROR] podman path is present but compose driver is not implemented in this template." >&2
  echo "        Tell me your environment (podman-compose? docker-compose?) and I’ll adapt it." >&2
  exit 2
else
  echo "[ERROR] Neither docker nor podman found on this node." >&2
  exit 2
fi

# Sanity check
if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
  echo "[ERROR] Built image not found in docker daemon: ${IMAGE_REF}" >&2
  echo "        Make sure docker-compose.yaml has: image: ${IMAGE_REPO}:\${TAG:-latest}" >&2
  exit 3
fi

# Make enroot image (.sqsh)
SQSH_PATH="${OUT_DIR}/${IMAGE_REPO//\//_}-${TAG}_${TIMESTAMP}.sqsh"

echo "[INFO] Creating enroot sqsh: ${SQSH_PATH}"
enroot import -o "${SQSH_PATH}" "dockerd://${IMAGE_REF}"

if [[ ! -s "${SQSH_PATH}" ]]; then
  echo "[ERROR] enroot import failed (sqsh not created): ${SQSH_PATH}" >&2
  exit 4
fi

# Check for import python module
echo "[INFO] Running import check inside enroot..."
IMPORT_STMT="import sys; print(sys.version)"
for m in "${IMPORT_CHECK_MODULES[@]}"; do
  IMPORT_STMT="${IMPORT_STMT}; import ${m}; print('${m}: OK')"
done

# Run with /workspace as working dir (matches your docker setup)
enroot start --mount "${REPO_ROOT}:/workspace" --mount "${REPO_ROOT}/.cache:/.cache" \
  "${SQSH_PATH}" \
  bash -lc "cd /workspace && python -c \"${IMPORT_STMT}\""

echo "=============================
