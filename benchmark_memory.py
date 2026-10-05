import argparse
import gc
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from axial_gru import (
    WunderAxialGRUModel,
    WunderLoopedAxialGRUModel,
    WunderReinjectedLoopedAxialGRUModel,
)
from wunder_dataset import TRAIN_PATH, load_sequence


MIB = 1024 ** 2


def build_model(model_name):
    if model_name == "baseline":
        return WunderAxialGRUModel(
            num_blocks=3,
        )

    if model_name == "looped":
        return WunderLoopedAxialGRUModel(
            num_iterations=3,
        )

    if model_name == "reinjected":
        return WunderReinjectedLoopedAxialGRUModel(
            num_iterations=3,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def parameter_bytes(model):
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if p.requires_grad
    )


def detach_hidden_states(hidden_states):
    if hidden_states is None:
        return None

    return [
        hidden.detach()
        if hidden is not None
        else None
        for hidden in hidden_states
    ]


def masked_mse(
    predictions,
    targets,
    mask,
):
    return F.mse_loss(
        predictions[mask],
        targets[mask],
    )


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_chunk(
    x,
    target,
    mask,
    start,
    chunk_size,
    device,
):
    end = start + chunk_size

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

    return (
        x_chunk,
        target_chunk,
        mask_chunk,
    )


@torch.no_grad()
def benchmark_inference(
    model_name,
    x,
    target,
    mask,
    chunk_size,
    device,
):
    clear_cuda()

    model = build_model(
        model_name
    ).to(device)

    model.eval()

    # Warm-up on the first chunk.
    x0, _, _ = get_chunk(
        x,
        target,
        mask,
        start=0,
        chunk_size=chunk_size,
        device=device,
    )

    _, hidden_states = model(
        x0,
        None,
        return_hidden=True
    )

    torch.cuda.synchronize()

    del x0
    gc.collect()

    # Measure the second chunk so that the
    # GRU receives a realistic previous
    # hidden state.
    torch.cuda.reset_peak_memory_stats(
        device
    )

    x1, _, _ = get_chunk(
        x,
        target,
        mask,
        start=chunk_size,
        chunk_size=chunk_size,
        device=device,
    )

    predictions, new_hidden_states = model(
        x1,
        hidden_states,
        return_hidden=True
    )

    torch.cuda.synchronize()

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

    # Keep references alive until after
    # memory statistics are collected.
    del predictions
    del new_hidden_states
    del hidden_states
    del x1
    del model

    clear_cuda()

    return {
        "peak_allocated_mib":
            peak_allocated,
        "peak_reserved_mib":
            peak_reserved,
    }


def benchmark_training(
    model_name,
    x,
    target,
    mask,
    chunk_size,
    device,
):
    clear_cuda()

    model = build_model(
        model_name
    ).to(device)

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # ----------------------------------
    # Warm-up training step
    #
    # This is important because Adam
    # creates its first- and second-
    # moment buffers lazily on the first
    # optimizer step.
    # ----------------------------------

    x0, target0, mask0 = get_chunk(
        x,
        target,
        mask,
        start=0,
        chunk_size=chunk_size,
        device=device,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    predictions, hidden_states = model(
        x0,
        None,
        return_hidden=True
    )

    loss = masked_mse(
        predictions,
        target0,
        mask0,
    )

    loss.backward()

    optimizer.step()

    hidden_states = (
        detach_hidden_states(
            hidden_states
        )
    )

    # Remove tensors belonging to the
    # warm-up computation graph.
    del predictions
    del loss
    del x0
    del target0
    del mask0

    optimizer.zero_grad(
        set_to_none=True
    )

    gc.collect()
    torch.cuda.synchronize()

    # ----------------------------------
    # Measured training step
    # ----------------------------------

    torch.cuda.reset_peak_memory_stats(
        device
    )

    x1, target1, mask1 = get_chunk(
        x,
        target,
        mask,
        start=chunk_size,
        chunk_size=chunk_size,
        device=device,
    )

    predictions, new_hidden_states = model(
        x1,
        hidden_states,
        return_hidden=True
    )

    loss = masked_mse(
        predictions,
        target1,
        mask1,
    )

    loss.backward()

    optimizer.step()

    torch.cuda.synchronize()

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

    del predictions
    del new_hidden_states
    del hidden_states
    del loss
    del x1
    del target1
    del mask1
    del optimizer
    del model

    clear_cuda()

    return {
        "peak_allocated_mib":
            peak_allocated,
        "peak_reserved_mib":
            peak_reserved,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--row-group",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "memory_benchmark.json"
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA runtime:",
        torch.version.cuda,
    )

    print(
        "Chunk size:",
        args.chunk_size,
    )

    # Load one real Wunder sequence.
    parquet_file = pq.ParquetFile(
        TRAIN_PATH
    )

    x, target, mask, seq_ix = (
        load_sequence(
            parquet_file,
            args.row_group,
        )
    )

    if len(x) < (
        2 * args.chunk_size
    ):
        raise ValueError(
            "Sequence is too short for "
            "two benchmark chunks."
        )

    print(
        "Sequence:",
        seq_ix,
    )

    print()

    results = []

    for model_name in (
        "baseline",
        "looped",
        "reinjected",
    ):
        print(
            "=" * 60
        )

        print(
            "Model:",
            model_name,
        )

        # Temporary CPU model only for
        # exact parameter statistics.
        cpu_model = build_model(
            model_name
        )

        params = count_parameters(
            cpu_model
        )

        param_mib = (
            parameter_bytes(
                cpu_model
            )
            / MIB
        )

        del cpu_model

        print(
            "Parameters:",
            params,
        )

        print(
            "Parameter memory:",
            f"{param_mib:.3f} MiB",
        )

        inference = (
            benchmark_inference(
                model_name,
                x,
                target,
                mask,
                args.chunk_size,
                device,
            )
        )

        print(
            "Inference peak allocated:",
            f"{inference['peak_allocated_mib']:.2f} MiB",
        )

        print(
            "Inference peak reserved:",
            f"{inference['peak_reserved_mib']:.2f} MiB",
        )

        training = (
            benchmark_training(
                model_name,
                x,
                target,
                mask,
                args.chunk_size,
                device,
            )
        )

        print(
            "Training peak allocated:",
            f"{training['peak_allocated_mib']:.2f} MiB",
        )

        print(
            "Training peak reserved:",
            f"{training['peak_reserved_mib']:.2f} MiB",
        )

        print()

        results.append({
            "model":
                model_name,
            "parameters":
                params,
            "parameter_memory_mib":
                param_mib,
            "inference_peak_allocated_mib":
                inference[
                    "peak_allocated_mib"
                ],
            "inference_peak_reserved_mib":
                inference[
                    "peak_reserved_mib"
                ],
            "training_peak_allocated_mib":
                training[
                    "peak_allocated_mib"
                ],
            "training_peak_reserved_mib":
                training[
                    "peak_reserved_mib"
                ],
        })

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "gpu":
            torch.cuda.get_device_name(0),
        "pytorch":
            torch.__version__,
        "cuda_runtime":
            torch.version.cuda,
        "chunk_size":
            args.chunk_size,
        "row_group":
            args.row_group,
        "seq_ix":
            seq_ix,
        "results":
            results,
    }

    with open(
        output_path,
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "=" * 60
    )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
