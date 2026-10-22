#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

for benchmark in math mmlu arc gpqa; do
  case "$benchmark" in
    math) input_file=valid.all.parquet ;;
    mmlu) input_file=valid.mmlu_pro.parquet ;;
    arc) input_file=valid.arc_c.parquet ;;
    gpqa) input_file=valid.gpqa.parquet ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/generate_vllm.py" \
    --model_path "$MODEL_PATH" \
    --input_file "$DATA/$input_file" \
    --remove_system True \
    --add_oat_evaluate True \
    --add_think_before_answer True \
    --output_file "$OUTPUT_DIR/${MODEL_NAME}_${benchmark}.jsonl" \
    --template "$TEMPLATE" "$@" > "$OUTPUT_DIR/${MODEL_NAME}_${benchmark}.log" 2>&1
done
