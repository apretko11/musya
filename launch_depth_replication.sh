#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Targeted depth replication
#
# Goal:
#   Extend selected depth configurations from seed 0
#   to seeds 0-4.
#
# Existing:
#   seed 0 for all configurations
#   depth 3 already has many seeds from main experiment
#
# New runs:
#
#   Baseline:
#       N = 1, 2, 4
#
#   Looped:
#       K = 2, 4
#
#   Reinjected:
#       K = 2, 4
#
#   Seeds:
#       1, 2, 3, 4
#
# K=1 looped/reinjected is not rerun because at one
# iteration it is equivalent to the one-block baseline.
# ============================================================


# ------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------

SPLIT_SEED=1234

TRAIN_SEQUENCES=100
VALID_SEQUENCES=20
EPOCHS=3

TRAIN_CHUNK_SIZE=512
EVAL_CHUNK_SIZE=8192

LEARNING_RATE=1e-4

BASE_RESULTS_DIR="results/weekend/depth"
BASE_LOG_DIR="logs/weekend/depth"

SEEDS=(
    1
    2
    3
    4
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}


evaluation_complete() {
    local evaluation_json="$1"

    if [[ ! -f "$evaluation_json" ]]; then
        return 1
    fi

    python - "$evaluation_json" <<'PY'
import json
import sys

path = sys.argv[1]

try:
    with open(path) as f:
        data = json.load(f)

    ok = (
        data.get("full_validation") is True
        and data.get("sequences") == 1873
        and data.get("chunk_size") == 8192
        and "weighted_pearson" in data
    )

except Exception:
    ok = False

raise SystemExit(0 if ok else 1)
PY
}


training_complete() {
    local checkpoint="$1"
    local results_json="$2"

    [[ -f "$checkpoint" && -f "$results_json" ]]
}


run_config() {
    local model="$1"
    local seed="$2"
    local config_name="$3"
    local num_blocks="$4"
    local num_iterations="$5"

    local output_dir
    output_dir="${BASE_RESULTS_DIR}/${config_name}"

    local log_dir
    log_dir="${BASE_LOG_DIR}/${config_name}"

    mkdir -p "$output_dir"
    mkdir -p "$output_dir/evaluations"
    mkdir -p "$log_dir"
    mkdir -p "$log_dir/evaluations"

    local run_name
    run_name="${model}_seed${seed}_split${SPLIT_SEED}"

    local checkpoint
    checkpoint="${output_dir}/${run_name}.pt"

    local results_json
    results_json="${output_dir}/${run_name}.json"

    local training_log
    training_log="${log_dir}/${run_name}.log"

    local evaluation_json
    evaluation_json="${output_dir}/evaluations/${run_name}_valid.json"

    local evaluation_log
    evaluation_log="${log_dir}/evaluations/${run_name}_valid.log"


    echo
    echo "============================================================"
    echo "[$(timestamp)]"
    echo "Configuration:  $config_name"
    echo "Model:          $model"
    echo "Seed:           $seed"
    echo "num_blocks:     $num_blocks"
    echo "num_iterations: $num_iterations"
    echo "============================================================"


    # --------------------------------------------------------
    # Whole run already complete?
    # --------------------------------------------------------

    if evaluation_complete "$evaluation_json"; then
        echo "Full evaluation already complete."
        echo "Skipping entire run:"
        echo "  $evaluation_json"
        return 0
    fi


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if training_complete "$checkpoint" "$results_json"; then
        echo "Training already complete."
        echo "Skipping training:"
        echo "  $checkpoint"
    else
        echo
        echo "Starting training..."

        python train_wunder.py \
            --model "$model" \
            --seed "$seed" \
            --split-seed "$SPLIT_SEED" \
            --train-sequences "$TRAIN_SEQUENCES" \
            --valid-sequences "$VALID_SEQUENCES" \
            --epochs "$EPOCHS" \
            --chunk-size "$TRAIN_CHUNK_SIZE" \
            --lr "$LEARNING_RATE" \
            --num-blocks "$num_blocks" \
            --num-iterations "$num_iterations" \
            --output-dir "$output_dir" \
            2>&1 | tee "$training_log"
    fi


    # --------------------------------------------------------
    # Checkpoint must exist before evaluation
    # --------------------------------------------------------

    if [[ ! -f "$checkpoint" ]]; then
        echo
        echo "ERROR: checkpoint missing:"
        echo "  $checkpoint"
        exit 1
    fi


    # --------------------------------------------------------
    # Full official Wunder validation
    # --------------------------------------------------------

    echo
    echo "Starting full validation..."

    python evaluate_wunder.py \
        --checkpoint "$checkpoint" \
        --chunk-size "$EVAL_CHUNK_SIZE" \
        --device cuda \
        --output-json "$evaluation_json" \
        2>&1 | tee "$evaluation_log"


    echo
    echo "[$(timestamp)] Completed:"
    echo "  configuration = $config_name"
    echo "  seed          = $seed"
}


# ============================================================
# Run experiments
# ============================================================

echo
echo "============================================================"
echo "Targeted depth replication"
echo "============================================================"
echo "Seeds:                 ${SEEDS[*]}"
echo "Training sequences:    $TRAIN_SEQUENCES"
echo "Validation sequences:  $VALID_SEQUENCES"
echo "Epochs:                $EPOCHS"
echo "Training chunk:        $TRAIN_CHUNK_SIZE"
echo "Evaluation chunk:      $EVAL_CHUNK_SIZE"
echo "============================================================"


for seed in "${SEEDS[@]}"; do

    # --------------------------------------------------------
    # Baseline independent depth
    # --------------------------------------------------------

    run_config \
        baseline \
        "$seed" \
        baseline_n1 \
        1 \
        3

    run_config \
        baseline \
        "$seed" \
        baseline_n2 \
        2 \
        3

    run_config \
        baseline \
        "$seed" \
        baseline_n4 \
        4 \
        3


    # --------------------------------------------------------
    # Looped recurrent depth
    # --------------------------------------------------------

    run_config \
        looped \
        "$seed" \
        looped_k2 \
        3 \
        2

    run_config \
        looped \
        "$seed" \
        looped_k4 \
        3 \
        4


    # --------------------------------------------------------
    # Reinjected recurrent depth
    # --------------------------------------------------------

    run_config \
        reinjected \
        "$seed" \
        reinjected_k2 \
        3 \
        2

    run_config \
        reinjected \
        "$seed" \
        reinjected_k4 \
        3 \
        4

done


echo
echo "============================================================"
echo "[$(timestamp)] Depth replication complete."
echo "============================================================"
