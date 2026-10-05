#!/usr/bin/env bash

set -euo pipefail

# Main experiment configuration
TRAIN_SEQUENCES=100
VALID_SEQUENCES=20
EPOCHS=3
SPLIT_SEED=1234
CHUNK_SIZE=512

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

# Keep training outputs and terminal logs separate.
mkdir -p results
mkdir -p logs/main_experiments

echo "========================================"
echo "Wunder main experiments"
echo "========================================"
echo "Train sequences: $TRAIN_SEQUENCES"
echo "Validation sequences: $VALID_SEQUENCES"
echo "Epochs: $EPOCHS"
echo "Split seed: $SPLIT_SEED"
echo "Chunk size: $CHUNK_SIZE"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "========================================"
echo

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        LOG_FILE="logs/main_experiments/${model}_seed${seed}_split${SPLIT_SEED}.log"

        echo
        echo "========================================"
        echo "Starting experiment"
        echo "Model: $model"
        echo "Seed: $seed"
        echo "Log: $LOG_FILE"
        echo "========================================"

        python train_wunder.py \
            --model "$model" \
            --seed "$seed" \
            --split-seed "$SPLIT_SEED" \
            --train-sequences "$TRAIN_SEQUENCES" \
            --valid-sequences "$VALID_SEQUENCES" \
            --epochs "$EPOCHS" \
            --chunk-size "$CHUNK_SIZE" \
            --output-dir results \
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
echo "All main experiments completed."
echo "========================================"
