import argparse
import json
import math
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from train_wunder import build_model
from wunder_dataset import (
    VALID_PATH,
    load_validation_sequence,
)


class RepresentationStats:
    """
    Aggregate statistics for latent tensors shaped:

        [B, T, A, D]

    Metrics are accumulated over all [B,T,A]
    latent vectors.
    """

    def __init__(self):
        self.vectors = 0
        self.elements = 0

        self.sum_cos_source = 0.0
        self.sum_sq_diff_source = 0.0
        self.sum_sq_source = 0.0

        self.sum_vector_norm = 0.0
        self.sum_sq_current = 0.0

        self.previous_vectors = 0
        self.sum_cos_previous = 0.0
        self.sum_sq_diff_previous = 0.0

    def update(
        self,
        current,
        source,
        previous=None,
    ):
        if current.shape != source.shape:
            raise ValueError(
                "current and source shapes differ: "
                f"{current.shape} vs {source.shape}"
            )

        d_model = current.shape[-1]

        current = (
            current.detach()
            .float()
            .reshape(-1, d_model)
        )

        source = (
            source.detach()
            .float()
            .reshape(-1, d_model)
        )

        num_vectors = current.shape[0]
        num_elements = current.numel()

        cosine_source = F.cosine_similarity(
            current,
            source,
            dim=-1,
            eps=1e-8,
        )

        diff_source = current - source

        self.vectors += num_vectors
        self.elements += num_elements

        self.sum_cos_source += (
            cosine_source
            .sum(dtype=torch.float64)
            .item()
        )

        self.sum_sq_diff_source += (
            diff_source
            .square()
            .sum(dtype=torch.float64)
            .item()
        )

        self.sum_sq_source += (
            source
            .square()
            .sum(dtype=torch.float64)
            .item()
        )

        self.sum_vector_norm += (
            torch.linalg.vector_norm(
                current,
                dim=-1,
            )
            .sum(dtype=torch.float64)
            .item()
        )

        self.sum_sq_current += (
            current
            .square()
            .sum(dtype=torch.float64)
            .item()
        )

        if previous is not None:
            if previous.shape != (
                current.shape[0],
                d_model,
            ):
                previous = (
                    previous.detach()
                    .float()
                    .reshape(-1, d_model)
                )

            cosine_previous = (
                F.cosine_similarity(
                    current,
                    previous,
                    dim=-1,
                    eps=1e-8,
                )
            )

            diff_previous = (
                current - previous
            )

            self.previous_vectors += (
                num_vectors
            )

            self.sum_cos_previous += (
                cosine_previous
                .sum(dtype=torch.float64)
                .item()
            )

            self.sum_sq_diff_previous += (
                diff_previous
                .square()
                .sum(dtype=torch.float64)
                .item()
            )

    def merge(self, other):
        self.vectors += other.vectors
        self.elements += other.elements

        self.sum_cos_source += (
            other.sum_cos_source
        )

        self.sum_sq_diff_source += (
            other.sum_sq_diff_source
        )

        self.sum_sq_source += (
            other.sum_sq_source
        )

        self.sum_vector_norm += (
            other.sum_vector_norm
        )

        self.sum_sq_current += (
            other.sum_sq_current
        )

        self.previous_vectors += (
            other.previous_vectors
        )

        self.sum_cos_previous += (
            other.sum_cos_previous
        )

        self.sum_sq_diff_previous += (
            other.sum_sq_diff_previous
        )

    def result(self):
        if self.vectors == 0:
            raise ValueError(
                "No vectors were accumulated."
            )

        rmse_source = math.sqrt(
            self.sum_sq_diff_source
            / self.elements
        )

        source_rms = math.sqrt(
            self.sum_sq_source
            / self.elements
        )

        current_rms = math.sqrt(
            self.sum_sq_current
            / self.elements
        )

        result = {
            "vectors":
                self.vectors,
            "mean_cosine_to_source":
                self.sum_cos_source
                / self.vectors,
            "rmse_to_source":
                rmse_source,
            "relative_rmse_to_source":
                (
                    rmse_source
                    / source_rms
                    if source_rms > 0
                    else None
                ),
            "source_rms":
                source_rms,
            "current_rms":
                current_rms,
            "mean_vector_l2_norm":
                self.sum_vector_norm
                / self.vectors,
        }

        if self.previous_vectors > 0:
            result.update({
                "mean_cosine_to_previous":
                    self.sum_cos_previous
                    / self.previous_vectors,
                "rmse_to_previous":
                    math.sqrt(
                        self.sum_sq_diff_previous
                        / self.elements
                    ),
            })

        return result


def load_config(
    checkpoint,
    results_json=None,
):
    if results_json is None:
        results_json = (
            checkpoint.with_suffix(".json")
        )

    if not results_json.is_file():
        raise FileNotFoundError(
            "Could not find training metadata: "
            f"{results_json}"
        )

    with results_json.open() as f:
        metadata = json.load(f)

    config = {
        "model":
            metadata["model"],
        "num_blocks":
            metadata["num_blocks"],
        "num_iterations":
            metadata[
                "num_iterations"
            ],
    }

    return (
        config,
        metadata,
        results_json,
    )


def model_depth(
    model_name,
    model,
):
    if model_name == "baseline":
        return len(model.blocks)

    return model.num_iterations


@torch.no_grad()
def analyze_sequence(
    model,
    model_name,
    x,
    device,
    chunk_size,
):
    depth = model_depth(
        model_name,
        model,
    )

    state_stats = [
        RepresentationStats()
        for _ in range(depth + 1)
    ]

    reinjection_stats = {
        iteration:
            RepresentationStats()
        for iteration in range(
            2,
            depth + 1,
        )
    } if model_name == "reinjected" else {}

    hidden_states = [
        None
        for _ in range(depth)
    ]

    gate = None

    if model_name == "reinjected":
        gate = torch.sigmoid(
            model.reinjection_logit
        )

    for start in range(
        0,
        len(x),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(x),
        )

        x_chunk = (
            x[start:end]
            .unsqueeze(0)
            .to(device)
        )

        # H0: original encoded observation.
        h0 = model.encoder(
            x_chunk
        )

        state_stats[0].update(
            current=h0,
            source=h0,
        )

        current = h0

        if model_name == "baseline":

            for index, block in enumerate(
                model.blocks
            ):
                previous_state = current

                current, new_hidden = block(
                    current,
                    hidden_states[index],
                )

                hidden_states[index] = (
                    new_hidden
                )

                state_stats[
                    index + 1
                ].update(
                    current=current,
                    source=h0,
                    previous=previous_state,
                )

        elif model_name == "looped":

            for index in range(
                model.num_iterations
            ):
                previous_state = current

                current, new_hidden = (
                    model.block(
                        current,
                        hidden_states[index],
                    )
                )

                hidden_states[index] = (
                    new_hidden
                )

                state_stats[
                    index + 1
                ].update(
                    current=current,
                    source=h0,
                    previous=previous_state,
                )

        elif model_name == "reinjected":

            for index in range(
                model.num_iterations
            ):
                previous_state = current

                block_input = current

                if index > 0:
                    block_input = (
                        (1.0 - gate)
                        * current
                        + gate
                        * h0
                    )

                    reinjection_stats[
                        index + 1
                    ].update(
                        current=block_input,
                        source=h0,
                        previous=current,
                    )

                current, new_hidden = (
                    model.block(
                        block_input,
                        hidden_states[index],
                    )
                )

                hidden_states[index] = (
                    new_hidden
                )

                # Compare post-block H_k
                # against both H0 and the
                # previous post-block state.
                state_stats[
                    index + 1
                ].update(
                    current=current,
                    source=h0,
                    previous=previous_state,
                )

        else:
            raise ValueError(
                f"Unknown model: "
                f"{model_name}"
            )

    return {
        "states":
            state_stats,
        "reinjection_inputs":
            reinjection_stats,
        "gate":
            (
                float(gate.item())
                if gate is not None
                else None
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--results-json",
        type=Path,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(VALID_PATH),
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.sequences <= 0:
        raise ValueError(
            "--sequences must be positive"
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be positive"
        )

    config, metadata, results_path = (
        load_config(
            args.checkpoint,
            args.results_json,
        )
    )

    device = (
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if args.device == "auto"
        else torch.device(args.device)
    )

    model = build_model(
        argparse.Namespace(
            **config
        )
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    depth = model_depth(
        config["model"],
        model,
    )

    global_states = [
        RepresentationStats()
        for _ in range(depth + 1)
    ]

    global_reinjection = {
        iteration:
            RepresentationStats()
        for iteration in range(
            2,
            depth + 1,
        )
    } if config[
        "model"
    ] == "reinjected" else {}

    per_sequence = []

    with pq.ParquetFile(
        args.data_path
    ) as parquet_file:

        total_sequences = (
            parquet_file
            .metadata
            .num_row_groups
        )

        count = min(
            args.sequences,
            total_sequences,
        )

        rng = random.Random(
            args.sample_seed
        )

        row_groups = sorted(
            rng.sample(
                range(total_sequences),
                count,
            )
        )

        for position, row_group in enumerate(
            row_groups,
            start=1,
        ):
            (
                x,
                _,
                _,
                _,
                seq_ix,
            ) = load_validation_sequence(
                parquet_file,
                row_group,
            )

            sequence_analysis = (
                analyze_sequence(
                    model=model,
                    model_name=config[
                        "model"
                    ],
                    x=x,
                    device=device,
                    chunk_size=args.chunk_size,
                )
            )

            sequence_states = (
                sequence_analysis[
                    "states"
                ]
            )

            for index, stats in enumerate(
                sequence_states
            ):
                global_states[
                    index
                ].merge(stats)

            sequence_reinjection = (
                sequence_analysis[
                    "reinjection_inputs"
                ]
            )

            for iteration, stats in (
                sequence_reinjection.items()
            ):
                global_reinjection[
                    iteration
                ].merge(stats)

            per_sequence.append({
                "row_group":
                    row_group,
                "seq_ix":
                    seq_ix,
                "states": [
                    {
                        "depth":
                            index,
                        **stats.result(),
                    }
                    for index, stats
                    in enumerate(
                        sequence_states
                    )
                ],
                "reinjection_inputs": [
                    {
                        "before_depth":
                            iteration,
                        **stats.result(),
                    }
                    for iteration, stats
                    in sorted(
                        sequence_reinjection.items()
                    )
                ],
            })

            print(
                f"  analyzed "
                f"{position}/{count} "
                f"(seq_ix={seq_ix})",
                flush=True,
            )

    aggregate_states = [
        {
            "depth":
                index,
            **stats.result(),
        }
        for index, stats
        in enumerate(global_states)
    ]

    aggregate_reinjection = [
        {
            "before_depth":
                iteration,
            **stats.result(),
        }
        for iteration, stats
        in sorted(
            global_reinjection.items()
        )
    ]

    gate = None

    if hasattr(
        model,
        "reinjection_logit",
    ):
        gate = float(
            torch.sigmoid(
                model.reinjection_logit
            ).item()
        )

    report = {
        "checkpoint":
            str(args.checkpoint),
        "results_json":
            str(results_path),
        "model_config":
            config,
        "training_metadata": {
            "seed":
                metadata.get("seed"),
            "train_sequences":
                metadata.get(
                    "train_sequences"
                ),
            "valid_sequences":
                metadata.get(
                    "valid_sequences"
                ),
            "best_epoch":
                metadata.get(
                    "best_epoch"
                ),
        },
        "device":
            str(device),
        "chunk_size":
            args.chunk_size,
        "sample_seed":
            args.sample_seed,
        "sequences":
            len(per_sequence),
        "reinjection_gate":
            gate,
        "aggregate_states":
            aggregate_states,
        "aggregate_reinjection_inputs":
            aggregate_reinjection,
        "per_sequence":
            per_sequence,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            allow_nan=False,
        )

        f.write("\n")

    print()
    print(
        "Latent-state summary"
    )

    print(
        "depth  "
        "cos(Hk,H0)  "
        "relRMSE(Hk,H0)  "
        "mean||Hk||  "
        "cos(Hk,Hprev)  "
        "RMSE(Hk,Hprev)"
    )

    for state in aggregate_states:
        print(
            f"{state['depth']:5d}  "
            f"{state['mean_cosine_to_source']:10.6f}  "
            f"{state['relative_rmse_to_source']:15.6f}  "
            f"{state['mean_vector_l2_norm']:10.6f}  "
            f"{state.get('mean_cosine_to_previous', float('nan')):13.6f}  "
            f"{state.get('rmse_to_previous', float('nan')):14.6f}"
        )

    if aggregate_reinjection:
        print()
        print(
            "Reinjection inputs"
        )

        print(
            "before  "
            "cos(Mk,H0)  "
            "relRMSE(Mk,H0)  "
            "RMSE(Mk,Hprev)"
        )

        for state in (
            aggregate_reinjection
        ):
            print(
                f"{state['before_depth']:6d}  "
                f"{state['mean_cosine_to_source']:10.6f}  "
                f"{state['relative_rmse_to_source']:15.6f}  "
                f"{state.get('rmse_to_previous', float('nan')):14.6f}"
            )

    if gate is not None:
        print()
        print(
            "Reinjection gate:",
            f"{gate:.8f}",
        )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
