import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from axial_gru import (
    WunderAxialGRUModel,
    WunderLoopedAxialGRUModel,
    WunderReinjectedLoopedAxialGRUModel,
    count_parameters,
)

from wunder_dataset import (
    TRAIN_PATH,
    load_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "looped",
            "reinjected",
        ],
        required=True,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=TRAIN_PATH,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Model/training random seed.",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=1234,
        help="Fixed seed controlling train/validation split.",
    )

    parser.add_argument(
        "--train-sequences",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--valid-sequences",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    if args.model == "baseline":
        return WunderAxialGRUModel(
            num_blocks=args.num_blocks,
        )

    if args.model == "looped":
        return WunderLoopedAxialGRUModel(
            num_iterations=args.num_iterations,
        )

    if args.model == "reinjected":
        return WunderReinjectedLoopedAxialGRUModel(
            num_iterations=args.num_iterations,
        )

    raise ValueError(
        f"Unknown model: {args.model}"
    )


def detach_hidden_states(hidden_states):
    if hidden_states is None:
        return None

    return [
        hidden.detach()
        for hidden in hidden_states
    ]


def train_sequence(
    model,
    optimizer,
    x,
    target,
    mask,
    device,
    chunk_size,
):
    model.train()

    hidden_states = None

    total_loss = 0.0
    total_prediction_steps = 0

    T = x.shape[0]

    for start in range(
        0,
        T,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            T,
        )

        x_chunk = (
            x[start:end]
            .unsqueeze(0)
            .to(device)
        )

        target_chunk = (
            target[start:end]
            .unsqueeze(0)
            .to(device)
        )

        mask_chunk = (
            mask[start:end]
            .unsqueeze(0)
            .to(device)
        )

        num_prediction_steps = (
            mask_chunk.sum().item()
        )

        optimizer.zero_grad()

        prediction, hidden_states = model(
            x_chunk,
            hidden_states=hidden_states,
            return_hidden=True,
        )

        hidden_states = detach_hidden_states(
            hidden_states
        )

        if num_prediction_steps == 0:
            continue

        loss = F.mse_loss(
            prediction[mask_chunk],
            target_chunk[mask_chunk],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * num_prediction_steps
        )

        total_prediction_steps += (
            num_prediction_steps
        )

    return (
        total_loss
        / total_prediction_steps
    )


def evaluate_sequence(
    model,
    x,
    target,
    mask,
    device,
    chunk_size,
):
    model.eval()

    hidden_states = None

    total_loss = 0.0
    total_prediction_steps = 0

    T = x.shape[0]

    with torch.no_grad():

        for start in range(
            0,
            T,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                T,
            )

            x_chunk = (
                x[start:end]
                .unsqueeze(0)
                .to(device)
            )

            target_chunk = (
                target[start:end]
                .unsqueeze(0)
                .to(device)
            )

            mask_chunk = (
                mask[start:end]
                .unsqueeze(0)
                .to(device)
            )

            prediction, hidden_states = model(
                x_chunk,
                hidden_states=hidden_states,
                return_hidden=True,
            )

            num_prediction_steps = (
                mask_chunk.sum().item()
            )

            if num_prediction_steps == 0:
                continue

            loss = F.mse_loss(
                prediction[mask_chunk],
                target_chunk[mask_chunk],
            )

            total_loss += (
                loss.item()
                * num_prediction_steps
            )

            total_prediction_steps += (
                num_prediction_steps
            )

    return (
        total_loss
        / total_prediction_steps
    )


def make_split(
    num_row_groups,
    train_sequences,
    valid_sequences,
    split_seed,
):
    required = (
        train_sequences
        + valid_sequences
    )

    if required > num_row_groups:
        raise ValueError(
            f"Requested {required} sequences, "
            f"but dataset only has "
            f"{num_row_groups} row groups."
        )

    indices = list(
        range(num_row_groups)
    )

    rng = random.Random(
        split_seed
    )

    rng.shuffle(indices)

    train_groups = indices[
        :train_sequences
    ]

    valid_groups = indices[
        train_sequences:required
    ]

    return (
        train_groups,
        valid_groups,
    )


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    pf = pq.ParquetFile(
        args.data_path
    )

    train_groups, valid_groups = make_split(
        num_row_groups=pf.metadata.num_row_groups,
        train_sequences=args.train_sequences,
        valid_sequences=args.valid_sequences,
        split_seed=args.split_seed,
    )

    model = build_model(
        args
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    parameter_count = count_parameters(
        model
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Parameters:",
        parameter_count,
    )

    print(
        "Training sequences:",
        len(train_groups),
    )

    print(
        "Validation sequences:",
        len(valid_groups),
    )

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = (
        f"{args.model}"
        f"_seed{args.seed}"
        f"_split{args.split_seed}"
    )

    checkpoint_path = (
        output_dir
        / f"{run_name}.pt"
    )

    results_path = (
        output_dir
        / f"{run_name}.json"
    )

    # --------------------------------------------------
    # Training bookkeeping
    # --------------------------------------------------

    history = []

    best_valid = float("inf")
    best_epoch = None

    start_time = time.time()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    for epoch in range(
        args.epochs
    ):
        print(
            f"\nEpoch "
            f"{epoch + 1}/"
            f"{args.epochs}"
        )

        train_losses = []

        for i, row_group in enumerate(
            train_groups,
            start=1,
        ):
            x, target, mask, seq_ix = (
                load_sequence(
                    pf,
                    row_group,
                )
            )

            loss = train_sequence(
                model=model,
                optimizer=optimizer,
                x=x,
                target=target,
                mask=mask,
                device=device,
                chunk_size=args.chunk_size,
            )

            train_losses.append(
                loss
            )

            print(
                f"  train "
                f"{i:3d}/"
                f"{len(train_groups)} "
                f"(seq_ix={seq_ix}): "
                f"{loss:.6f}"
            )

        train_mean = float(
            np.mean(train_losses)
        )

        print(
            "Train mean loss:",
            f"{train_mean:.6f}",
        )

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        valid_losses = []

        for i, row_group in enumerate(
            valid_groups,
            start=1,
        ):
            x, target, mask, seq_ix = (
                load_sequence(
                    pf,
                    row_group,
                )
            )

            loss = evaluate_sequence(
                model=model,
                x=x,
                target=target,
                mask=mask,
                device=device,
                chunk_size=args.chunk_size,
            )

            valid_losses.append(
                loss
            )

            print(
                f"  valid "
                f"{i:3d}/"
                f"{len(valid_groups)} "
                f"(seq_ix={seq_ix}): "
                f"{loss:.6f}"
            )

        valid_mean = float(
            np.mean(valid_losses)
        )

        print(
            "Validation mean loss:",
            f"{valid_mean:.6f}",
        )

        # ----------------------------------------------
        # Save best validation checkpoint
        # ----------------------------------------------

        if valid_mean < best_valid:
            best_valid = valid_mean
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"New best validation loss: "
                f"{best_valid:.6f}"
            )

        epoch_result = {
            "epoch": epoch + 1,
            "train_mse": train_mean,
            "valid_mse": valid_mean,
        }

        history.append(
            epoch_result
        )

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------

    elapsed_seconds = (
        time.time() - start_time
    )

    print(
        "\nElapsed time:",
        f"{elapsed_seconds:.2f} seconds",
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation MSE:",
        f"{best_valid:.6f}",
    )

    # --------------------------------------------------
    # Reload the best model
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    # --------------------------------------------------
    # Inspect reinjection gate, if present
    # --------------------------------------------------

    reinjection_gate = None

    if hasattr(
        model,
        "reinjection_logit",
    ):
        reinjection_gate = float(
            torch.sigmoid(
                model.reinjection_logit
            ).item()
        )

        print(
            "\nLearned reinjection gate:",
            reinjection_gate,
        )

    # --------------------------------------------------
    # Save experiment metadata/results
    # --------------------------------------------------

    results = {
        "model": args.model,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_sequences": (
            args.train_sequences
        ),
        "valid_sequences": (
            args.valid_sequences
        ),
        "epochs": args.epochs,
        "chunk_size": args.chunk_size,
        "learning_rate": args.lr,
        "num_blocks": args.num_blocks,
        "num_iterations": (
            args.num_iterations
        ),
        "parameters": parameter_count,
        "device": str(device),
        "train_row_groups": (
            train_groups
        ),
        "valid_row_groups": (
            valid_groups
        ),
        "best_epoch": best_epoch,
        "best_valid_mse": best_valid,
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "reinjection_gate": (
            reinjection_gate
        ),
        "history": history,
    }

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nSaved best checkpoint:",
        checkpoint_path,
    )

    print(
        "Saved results:",
        results_path,
    )


if __name__ == "__main__":
    main()
