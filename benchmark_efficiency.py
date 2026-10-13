import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from axial_gru import count_parameters
from train_wunder import build_model
from wunder_dataset import (
    TRAIN_PATH,
    load_sequence,
)


MIB = 1024 ** 2


def parameter_bytes(model):
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if p.requires_grad
    )


def detach_hidden_states(
    hidden_states,
):
    if hidden_states is None:
        return None

    return [
        hidden.detach()
        for hidden in hidden_states
    ]


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_model(args, device):
    model_args = argparse.Namespace(
        model=args.model,
        num_blocks=args.num_blocks,
        num_iterations=args.num_iterations,
    )

    model = build_model(
        model_args
    )

    if args.checkpoint is not None:
        state_dict = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    return model.to(device)


def warm_up_training(
    model,
    optimizer,
    x,
    target,
    mask,
    device,
    chunk_size,
):
    model.train()

    end = min(
        chunk_size,
        len(x),
    )

    x_chunk = (
        x[:end]
        .unsqueeze(0)
        .to(device)
    )

    target_chunk = (
        target[:end]
        .unsqueeze(0)
        .to(device)
    )

    mask_chunk = (
        mask[:end]
        .unsqueeze(0)
        .to(device)
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    prediction, _ = model(
        x_chunk,
        hidden_states=None,
        return_hidden=True,
    )

    if mask_chunk.any():
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

    torch.cuda.synchronize()


def benchmark_training_once(
    args,
    x,
    target,
    mask,
    device,
):
    clear_cuda()

    model = make_model(
        args,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # Allocate Adam state and warm kernels.
    warm_up_training(
        model=model,
        optimizer=optimizer,
        x=x,
        target=target,
        mask=mask,
        device=device,
        chunk_size=args.train_chunk_size,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    hidden_states = None

    start_time = time.perf_counter()

    for start in range(
        0,
        len(x),
        args.train_chunk_size,
    ):
        end = min(
            start
            + args.train_chunk_size,
            len(x),
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

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction, hidden_states = (
            model(
                x_chunk,
                hidden_states=hidden_states,
                return_hidden=True,
            )
        )

        hidden_states = (
            detach_hidden_states(
                hidden_states
            )
        )

        if not mask_chunk.any():
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

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
        / MIB
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
        / MIB
    )

    del optimizer
    del model
    del hidden_states

    clear_cuda()

    return {
        "seconds":
            elapsed,
        "peak_allocated_mib":
            peak_allocated,
        "peak_reserved_mib":
            peak_reserved,
    }


@torch.no_grad()
def benchmark_inference_once(
    args,
    x,
    device,
):
    clear_cuda()

    model = make_model(
        args,
        device,
    )

    model.eval()

    # Warm-up.
    warm_end = min(
        args.eval_chunk_size,
        len(x),
    )

    warm_chunk = (
        x[:warm_end]
        .unsqueeze(0)
        .to(device)
    )

    model(
        warm_chunk,
        hidden_states=None,
        return_hidden=True,
    )

    torch.cuda.synchronize()

    del warm_chunk

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    hidden_states = None

    start_time = time.perf_counter()

    for start in range(
        0,
        len(x),
        args.eval_chunk_size,
    ):
        end = min(
            start
            + args.eval_chunk_size,
            len(x),
        )

        x_chunk = (
            x[start:end]
            .unsqueeze(0)
            .to(device)
        )

        _, hidden_states = model(
            x_chunk,
            hidden_states=hidden_states,
            return_hidden=True,
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
        / MIB
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
        / MIB
    )

    del model
    del hidden_states

    clear_cuda()

    return {
        "seconds":
            elapsed,
        "peak_allocated_mib":
            peak_allocated,
        "peak_reserved_mib":
            peak_reserved,
    }


def summarize(values):
    result = {
        "mean":
            statistics.mean(values),
        "min":
            min(values),
        "max":
            max(values),
    }

    if len(values) > 1:
        result["std"] = (
            statistics.stdev(values)
        )
    else:
        result["std"] = 0.0

    return result


def main():
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
        "--checkpoint",
        type=Path,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(TRAIN_PATH),
    )

    parser.add_argument(
        "--row-group",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--train-chunk-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    if args.repetitions <= 0:
        raise ValueError(
            "--repetitions must be positive"
        )

    device = torch.device(
        "cuda"
    )

    with pq.ParquetFile(
        args.data_path
    ) as parquet_file:
        (
            x,
            target,
            mask,
            seq_ix,
        ) = load_sequence(
            parquet_file,
            args.row_group,
        )

    cpu_model = make_model(
        args,
        torch.device("cpu"),
    )

    parameters = count_parameters(
        cpu_model
    )

    parameter_memory_mib = (
        parameter_bytes(
            cpu_model
        )
        / MIB
    )

    del cpu_model

    training_runs = []
    inference_runs = []

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Configuration:",
        args.model,
        "num_blocks=",
        args.num_blocks,
        "num_iterations=",
        args.num_iterations,
    )

    print(
        "Parameters:",
        parameters,
    )

    for repetition in range(
        1,
        args.repetitions + 1,
    ):
        print(
            f"\nRepetition "
            f"{repetition}/"
            f"{args.repetitions}"
        )

        training = (
            benchmark_training_once(
                args,
                x,
                target,
                mask,
                device,
            )
        )

        inference = (
            benchmark_inference_once(
                args,
                x,
                device,
            )
        )

        training_runs.append(
            training
        )

        inference_runs.append(
            inference
        )

        print(
            "  train:",
            f"{training['seconds']:.4f}s",
            f"{training['peak_allocated_mib']:.2f} MiB allocated",
            f"{training['peak_reserved_mib']:.2f} MiB reserved",
        )

        print(
            "  infer:",
            f"{inference['seconds']:.4f}s",
            f"{inference['peak_allocated_mib']:.2f} MiB allocated",
            f"{inference['peak_reserved_mib']:.2f} MiB reserved",
        )

    report = {
        "gpu":
            torch.cuda.get_device_name(0),
        "pytorch":
            torch.__version__,
        "cuda_runtime":
            torch.version.cuda,
        "model":
            args.model,
        "num_blocks":
            args.num_blocks,
        "num_iterations":
            args.num_iterations,
        "checkpoint":
            (
                str(args.checkpoint)
                if args.checkpoint
                is not None
                else None
            ),
        "row_group":
            args.row_group,
        "seq_ix":
            seq_ix,
        "parameters":
            parameters,
        "parameter_memory_mib":
            parameter_memory_mib,
        "train_chunk_size":
            args.train_chunk_size,
        "eval_chunk_size":
            args.eval_chunk_size,
        "repetitions":
            args.repetitions,
        "training_runs":
            training_runs,
        "inference_runs":
            inference_runs,
        "training_summary": {
            "seconds":
                summarize([
                    run["seconds"]
                    for run
                    in training_runs
                ]),
            "peak_allocated_mib":
                summarize([
                    run[
                        "peak_allocated_mib"
                    ]
                    for run
                    in training_runs
                ]),
            "peak_reserved_mib":
                summarize([
                    run[
                        "peak_reserved_mib"
                    ]
                    for run
                    in training_runs
                ]),
        },
        "inference_summary": {
            "seconds":
                summarize([
                    run["seconds"]
                    for run
                    in inference_runs
                ]),
            "peak_allocated_mib":
                summarize([
                    run[
                        "peak_allocated_mib"
                    ]
                    for run
                    in inference_runs
                ]),
            "peak_reserved_mib":
                summarize([
                    run[
                        "peak_reserved_mib"
                    ]
                    for run
                    in inference_runs
                ]),
        },
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
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
