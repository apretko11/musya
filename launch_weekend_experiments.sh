#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Weekend Wunder experiments
#
# Usage:
#
#   PHASE=extra_seeds ./launch_weekend_experiments.sh
#
#   PHASE=depth_seed0 ./launch_weekend_experiments.sh
#
#   PHASE=all ./launch_weekend_experiments.sh
#
# Default:
#
#   PHASE=extra_seeds
#
# The script is resumable:
# - completed training runs are skipped
# - completed full-validation evaluations are skipped
# ============================================================


# ------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------

PHASE="${PHASE:-extra_seeds}"

SPLIT_SEED=1234

TRAIN_SEQUENCES=100
VALID_SEQUENCES=20
EPOCHS=3

TRAIN_CHUNK_SIZE=512
EVAL_CHUNK_SIZE=8192

LEARNING_RATE=1e-4

BASE_RESULTS_DIR="results/weekend"
BASE_LOG_DIR="logs/weekend"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}


training_complete() {
    local checkpoint="$1"
    local results_json="$2"

    [[ -f "$checkpoint" && -f "$results_json" ]]
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


run_training_and_evaluation() {
    local model="$1"
    local seed="$2"
    local output_dir="$3"
    local log_dir="$4"
    local num_blocks="$5"
    local num_iterations="$6"

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

    # --------------------------------------------------------
    # If a complete official evaluation already exists,
    # the entire run is finished. This allows completed runs
    # to be resumed/skipped even on another machine where
    # the ignored .pt checkpoint is not present.
    # --------------------------------------------------------

    if evaluation_complete \
        "$evaluation_json"
    then
        echo
        echo "Full evaluation already complete."
        echo "Skipping entire run:"
        echo "  $evaluation_json"
        return 0
    fi

    echo
    echo "============================================================"
    echo "[$(timestamp)]"
    echo "Model:          $model"
    echo "Seed:           $seed"
    echo "num_blocks:     $num_blocks"
    echo "num_iterations: $num_iterations"
    echo "Output dir:     $output_dir"
    echo "============================================================"


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if training_complete \
        "$checkpoint" \
        "$results_json"
    then
        echo
        echo "Training already complete."
        echo "Skipping:"
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

        echo
        echo "Training finished."
    fi


    # --------------------------------------------------------
    # Verify checkpoint exists
    # --------------------------------------------------------

    if [[ ! -f "$checkpoint" ]]; then
        echo
        echo "ERROR:"
        echo "Checkpoint does not exist:"
        echo "  $checkpoint"
        exit 1
    fi


    # --------------------------------------------------------
    # Official full validation evaluation
    # --------------------------------------------------------

    if evaluation_complete \
        "$evaluation_json"
    then
        echo
        echo "Full evaluation already complete."
        echo "Skipping:"
        echo "  $evaluation_json"

    else
        echo
        echo "Starting full official validation..."
        echo "Evaluation chunk size: $EVAL_CHUNK_SIZE"

        python evaluate_wunder.py \
            --checkpoint "$checkpoint" \
            --chunk-size "$EVAL_CHUNK_SIZE" \
            --device cuda \
            --output-json "$evaluation_json" \
            2>&1 | tee "$evaluation_log"

        echo
        echo "Evaluation finished."
    fi


    echo
    echo "[$(timestamp)] Run complete:"
    echo "  model = $model"
    echo "  seed  = $seed"
}


# ============================================================
# Phase A:
# Extend main experiment from 3 seeds to 10 seeds
#
# Existing:
#   seeds 0,1,2
#
# Add:
#   seeds 3,4,5,6,7,8,9
# ============================================================

run_extra_seeds() {

    echo
    echo "############################################################"
    echo "# PHASE A: MAIN COMPARISON — EXTRA SEEDS"
    echo "############################################################"
    echo

    local output_dir
    output_dir="${BASE_RESULTS_DIR}/extra_seeds"

    local log_dir
    log_dir="${BASE_LOG_DIR}/extra_seeds"

    local models=(
        baseline
        looped
        reinjected
    )

    local seeds=(
        3
        4
        5
        6
        7
        8
        9
    )

    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do

            run_training_and_evaluation \
                "$model" \
                "$seed" \
                "$output_dir" \
                "$log_dir" \
                3 \
                3

        done
    done
}


# ============================================================
# Phase B:
# Initial depth sweep using seed 0
#
# Existing depth-3 result already exists in:
#
#   results/
#
# so only run depths:
#
#   1
#   2
#   4
#
# Baseline:
#   num_blocks = depth
#
# Looped / Reinjected:
#   num_iterations = depth
# ============================================================

run_depth_seed0() {

    echo
    echo "############################################################"
    echo "# PHASE B: DEPTH SWEEP — SEED 0"
    echo "############################################################"
    echo

    local seed=0

    local depths=(
        1
        2
        4
    )


    # --------------------------------------------------------
    # Baseline depth
    # --------------------------------------------------------

    for depth in "${depths[@]}"; do

        local output_dir
        output_dir="${BASE_RESULTS_DIR}/depth/baseline_n${depth}"

        local log_dir
        log_dir="${BASE_LOG_DIR}/depth/baseline_n${depth}"

        run_training_and_evaluation \
            baseline \
            "$seed" \
            "$output_dir" \
            "$log_dir" \
            "$depth" \
            3

    done


    # --------------------------------------------------------
    # Looped recurrent depth
    # --------------------------------------------------------

    for depth in "${depths[@]}"; do

        local output_dir
        output_dir="${BASE_RESULTS_DIR}/depth/looped_k${depth}"

        local log_dir
        log_dir="${BASE_LOG_DIR}/depth/looped_k${depth}"

        run_training_and_evaluation \
            looped \
            "$seed" \
            "$output_dir" \
            "$log_dir" \
            3 \
            "$depth"

    done


    # --------------------------------------------------------
    # Reinjected recurrent depth
    # --------------------------------------------------------

    for depth in "${depths[@]}"; do

        local output_dir
        output_dir="${BASE_RESULTS_DIR}/depth/reinjected_k${depth}"

        local log_dir
        log_dir="${BASE_LOG_DIR}/depth/reinjected_k${depth}"

        run_training_and_evaluation \
            reinjected \
            "$seed" \
            "$output_dir" \
            "$log_dir" \
            3 \
            "$depth"

    done
}


# ============================================================
# Main
# ============================================================

echo
echo "============================================================"
echo "Wunder weekend experiments"
echo "============================================================"
echo "Phase:                 $PHASE"
echo "Train sequences:       $TRAIN_SEQUENCES"
echo "Internal valid:        $VALID_SEQUENCES"
echo "Epochs:                $EPOCHS"
echo "Training chunk size:   $TRAIN_CHUNK_SIZE"
echo "Evaluation chunk size: $EVAL_CHUNK_SIZE"
echo "Split seed:            $SPLIT_SEED"
echo "============================================================"


case "$PHASE" in

    extra_seeds)
        run_extra_seeds
        ;;

    depth_seed0)
        run_depth_seed0
        ;;

    all)
        run_extra_seeds
        run_depth_seed0
        ;;

    *)
        echo
        echo "Unknown PHASE:"
        echo "  $PHASE"
        echo
        echo "Valid choices:"
        echo "  extra_seeds"
        echo "  depth_seed0"
        echo "  all"
        exit 1
        ;;

esac


echo
echo "============================================================"
echo "[$(timestamp)] Requested phase completed successfully."
echo "============================================================"
