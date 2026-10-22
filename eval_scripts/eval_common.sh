#!/usr/bin/env bash
# Shared configuration for evaluation entry points.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA:?Set DATA to the benchmark parquet directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the evaluation output directory}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
MODEL_NAME="${MODEL_NAME:-$(basename -- "${MODEL_PATH%/}")}"
GPU="${GPU:-0,1,2,3}"
PYTHON="${PYTHON:-python}"

# Resolve caller-relative paths before switching to the codes directory.
for variable in DATA OUTPUT_DIR MODEL_PATH; do
  value="${!variable}"
  if [[ "$value" != /* ]]; then
    printf -v "$variable" '%s/%s' "$PWD" "$value"
  fi
done

if [[ -z "${TEMPLATE:-}" ]]; then
  if [[ "$MODEL_NAME" == "eurus-2-7b-prime-zero" ]]; then
    TEMPLATE=prime
  elif [[ "$MODEL_NAME" == "simple-rl-zero" ]]; then
    TEMPLATE=qwen
  else
    TEMPLATE=own
  fi
fi
mkdir -p -- "$OUTPUT_DIR"
cd -- "$SCRIPT_DIR/.."
