#!/usr/bin/env bash
set +x
set -euo pipefail

_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
_CALLER_DIR="$PWD"
CODES_ROOT="$(cd -- "$_COMMON_DIR/.." && pwd -P)"
export CODES_ROOT

caller_path() {
    if [[ "$1" == /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$_CALLER_DIR" "$1"
    fi
}

export PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
if [[ "$PYTHON_BIN" == */* && "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="$(caller_path "$PYTHON_BIN")"
fi
export VERL_ROOT="$(caller_path "${VERL_ROOT:-$CODES_ROOT/luffy/verl}")"
export PYTHONPATH="$VERL_ROOT:$CODES_ROOT/luffy${PYTHONPATH:+:$PYTHONPATH}"

hydra_string() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "$value"
}

init_experiment() {
    export EXP_NAME="${EXP_NAME:-$1}"
    export WANDB_PROJECT="${WANDB_PROJECT:-$2}"
    export DATA_DIR="$(caller_path "${DATA_DIR:-${DATA:-$CODES_ROOT/data}}")"
    export MODEL_PATH="$(caller_path "${MODEL_PATH:-$CODES_ROOT/checkpoints/$3}")"
    export OUTPUT_DIR="$(caller_path "${OUTPUT_DIR:-$CODES_ROOT/rl_results/$EXP_NAME}")"
    export LOG_DIR="$(caller_path "${LOG_DIR:-$OUTPUT_DIR}")"
    export WANDB_DIR="$(caller_path "${WANDB_DIR:-$OUTPUT_DIR}")"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONUNBUFFERED=1
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

    if [[ "$WANDB_MODE" == online && -z "${WANDB_API_KEY:-}" ]]; then
        printf '%s\n' 'Online logging requires WANDB_API_KEY in the environment.' >&2
        exit 1
    fi
    case "${STOP_RAY:-0}" in
        0|1) ;;
        *) printf '%s\n' 'STOP_RAY must be 0 or 1.' >&2; exit 1 ;;
    esac

    mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR" "$WANDB_DIR"
    cd -- "$VERL_ROOT"
    if [[ "${STOP_RAY:-0}" == 1 ]]; then
        ray stop
    fi
}

run_training() {
    local -a statuses
    if "$PYTHON_BIN" "$@" 2>&1 | tee -a "$LOG_DIR/run.log"; then
        return 0
    else
        statuses=("${PIPESTATUS[@]}")
        if (( statuses[0] != 0 )); then
            return "${statuses[0]}"
        fi
        return "${statuses[1]}"
    fi
}
