#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

# Preserve the existing local code-execution integration and allow overrides.
export LUFFY_CPP_COMPILER="${LUFFY_CPP_COMPILER:-g++}"
export LUFFY_CODE_COMPILE_TIMEOUT="${LUFFY_CODE_COMPILE_TIMEOUT:-20}"
# RUN_TIMEOUT limits CPU time; the wall deadline includes WALL_MULTIPLIER.
export LUFFY_CODE_RUN_TIMEOUT="${LUFFY_CODE_RUN_TIMEOUT:-10}"
export LUFFY_CODE_WALL_MULTIPLIER="${LUFFY_CODE_WALL_MULTIPLIER:-60}"
export LUFFY_CODE_MEMORY_LIMIT_MB="${LUFFY_CODE_MEMORY_LIMIT_MB:-1024}"
export LUFFY_CODE_OPT_LEVEL="${LUFFY_CODE_OPT_LEVEL:--O2}"
export LUFFY_CODE_CACHE_SIZE="${LUFFY_CODE_CACHE_SIZE:-50000}"
export LUFFY_REWARD_WORKERS="${LUFFY_REWARD_WORKERS:-32}"

# LiveCodeBench uses the original public tests and one sample per input row.
# Generated programs run locally and are untrusted; this is NOT a sandbox.
for benchmark in livecodebench arc gpqa mmlu; do
  args=(--remove_system True --add_think_before_answer True)
  case "$benchmark" in
    livecodebench) input_file=livecodebench.parquet ;;
    arc) input_file=valid.arc_c.parquet; args+=(--add_oat_evaluate True) ;;
    gpqa) input_file=valid.gpqa.parquet; args+=(--add_oat_evaluate True) ;;
    mmlu) input_file=valid.mmlu_pro.parquet; args+=(--add_oat_evaluate True) ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/generate_vllm.py" \
    --model_path "$MODEL_PATH" \
    --input_file "$DATA/$input_file" \
    "${args[@]}" \
    --output_file "$OUTPUT_DIR/${MODEL_NAME}_${benchmark}.jsonl" \
    --template "$TEMPLATE" "$@" > "$OUTPUT_DIR/${MODEL_NAME}_${benchmark}.log" 2>&1
done

# Optional pass@k remains disabled. Plain input reports only pass@1.
# CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/pass_at_k_vllm.py" \
#   --model_path "$MODEL_PATH" \
#   --input_file "$DATA/livecodebench.parquet" \
#   --remove_system True \
#   --add_think_before_answer True \
#   --output_file "$OUTPUT_DIR/${MODEL_NAME}_livecodebench_pass_at_k.jsonl" \
#   --template "$TEMPLATE" > "$OUTPUT_DIR/${MODEL_NAME}_livecodebench_pass_at_k.log" 2>&1
