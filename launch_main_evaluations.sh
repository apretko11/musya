#!/usr/bin/env bash

set -euo pipefail

SPLIT_SEED=1234

MODELS=(
    baseline
    looped
    reinjected
)

SEEDS=(
    0
    1
    2
)

mkdir -p results/evaluations
mkdir -p logs/main_evaluations

echo "========================================"
echo "Wunder full validation evaluations"
echo "========================================"
echo

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        CHECKPOINT="results/${model}_seed${seed}_split${SPLIT_SEED}.pt"
        OUTPUT_JSON="results/evaluations/${model}_seed${seed}_split${SPLIT_SEED}_valid.json"
        LOG_FILE="logs/main_evaluations/${model}_seed${seed}_split${SPLIT_SEED}.log"

        echo
        echo "========================================"
        echo "Evaluating"
        echo "Model:      $model"
        echo "Seed:       $seed"
        echo "Checkpoint: $CHECKPOINT"
        echo "Output:     $OUTPUT_JSON"
        echo "Log:        $LOG_FILE"
        echo "========================================"

        python evaluate_wunder.py \
            --checkpoint "$CHECKPOINT" \
            --device cuda \
            --output-json "$OUTPUT_JSON" \
            2>&1 | tee "$LOG_FILE"

        echo
        echo "Finished:"
        echo "  model = $model"
        echo "  seed  = $seed"
        echo
    done
done

echo
echo "========================================"
echo "All full validation evaluations completed."
echo "========================================"
