#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

# Preserve the original 256 replicated rows per problem in each input.
for SET in aime aime25; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/pass_at_k_vllm.py" \
    --model_path "$MODEL_PATH" \
    --input_file "$DATA/valid_${SET}_256.parquet" \
    --remove_system True \
    --add_think_before_answer True \
    --output_file "$OUTPUT_DIR/${MODEL_NAME}_${SET}_pass_at_k.jsonl" \
    --template "$TEMPLATE" "$@" > "$OUTPUT_DIR/${MODEL_NAME}_${SET}_pass_at_k.log" 2>&1
done
