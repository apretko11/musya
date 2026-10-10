#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT="results/baseline_seed0_split1234.pt"

MAX_SEQUENCES=100

CHUNK_SIZES=(
    512
    2048
    8192
    20000
)

OUTPUT_DIR="results/eval_chunk_benchmark"
LOG_DIR="logs/eval_chunk_benchmark"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Evaluation chunk-size benchmark"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Sequences:  $MAX_SEQUENCES"
echo

for chunk_size in "${CHUNK_SIZES[@]}"; do

    OUTPUT_JSON="$OUTPUT_DIR/chunk_${chunk_size}.json"
    LOG_FILE="$LOG_DIR/chunk_${chunk_size}.log"

    echo
    echo "========================================"
    echo "Chunk size: $chunk_size"
    echo "========================================"

    python evaluate_wunder.py \
        --checkpoint "$CHECKPOINT" \
        --chunk-size "$chunk_size" \
        --max-sequences "$MAX_SEQUENCES" \
        --device cuda \
        --output-json "$OUTPUT_JSON" \
        2>&1 | tee "$LOG_FILE"

done

echo
echo "========================================"
echo "Chunk benchmark complete"
echo "========================================"
