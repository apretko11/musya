"""Evaluate a trained checkpoint on complete official Wunder validation sequences.

Example (automatically reads the checkpoint's sibling training-results JSON)::

    python evaluate_wunder.py --checkpoint results/baseline_seed0_split1234.pt \
        --max-sequences 2 --output-json results/baseline_valid_smoke.json

Without training metadata, supply --model and --num-blocks (baseline) or
--num-iterations (looped/reinjected). Other dimensions match train_wunder.py.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from train_wunder import build_model
from wunder_dataset import VALID_PATH, load_validation_sequence
from wunder_metric import GlobalAccumulator


SEQUENCE_LENGTH = 20000
MODEL_NAMES = ("baseline", "looped", "reinjected")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--results-json", type=Path,
        help="Training results; defaults to the checkpoint path with .json suffix.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--num-blocks", type=positive_int)
    parser.add_argument("--num-iterations", type=positive_int)
    parser.add_argument("--data-path", type=Path, default=Path(VALID_PATH))
    parser.add_argument(
        "--chunk-size", type=positive_int,
        help="Defaults to training metadata's chunk_size, or 512.",
    )
    parser.add_argument("--max-sequences", type=positive_int)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--output-json", type=Path,
        help="Defaults to <checkpoint stem>_validation.json beside the checkpoint.",
    )
    return parser.parse_args(argv)


def resolve_config(args):
    results_path = args.results_json
    if results_path is None:
        sibling = args.checkpoint.with_suffix(".json")
        if sibling.is_file():
            results_path = sibling
    metadata = {}
    if results_path is not None:
        with results_path.open() as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError("Training results JSON must contain an object")

    config = {}
    for key in ("model", "num_blocks", "num_iterations"):
        supplied = getattr(args, key)
        saved = metadata.get(key)
        if supplied is not None and saved is not None and supplied != saved:
            raise ValueError(f"--{key.replace('_', '-')} conflicts with {results_path}")
        config[key] = saved if saved is not None else supplied

    if config["model"] not in MODEL_NAMES:
        raise ValueError("Supply --model or a training results JSON containing model")
    depth_key = "num_blocks" if config["model"] == "baseline" else "num_iterations"
    depth = config[depth_key]
    if type(depth) is not int or depth <= 0:
        raise ValueError(
            f"Supply --{depth_key.replace('_', '-')} or a training results JSON "
            f"containing a positive {depth_key}; depth cannot be inferred safely "
            "from shared weights"
        )
    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = metadata.get("chunk_size", 512)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return config, chunk_size, results_path


@torch.no_grad()
def predict_sequence(model, x, device, chunk_size):
    """Carry each depth's temporal state across chunks; reset on every call."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    model.eval()
    hidden_states = None
    predictions = torch.empty((len(x), 2), dtype=torch.float32)
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        x_chunk = x[start:end].unsqueeze(0).to(device)
        prediction, hidden_states = model(
            x_chunk, hidden_states=hidden_states, return_hidden=True,
        )
        if prediction.shape != (1, end - start, 2):
            raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
        predictions[start:end] = prediction.squeeze(0).cpu()
    return predictions


def evaluate(model, parquet_file, device, chunk_size, max_sequences=None):
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
    count = parquet_file.metadata.num_row_groups
    if max_sequences is not None:
        count = min(count, max_sequences)
    accumulator = GlobalAccumulator()
    for row_group in range(count):
        x, target, need_prediction, is_scored, seq_ix = load_validation_sequence(
            parquet_file, row_group,
        )
        if x.shape != (SEQUENCE_LENGTH, 2, 60):
            raise ValueError(f"Sequence {seq_ix}: expected [20000, 2, 60], got {tuple(x.shape)}")
        if target.shape != (SEQUENCE_LENGTH, 2):
            raise ValueError(f"Sequence {seq_ix}: expected [20000, 2] targets")
        if need_prediction.shape != (SEQUENCE_LENGTH,) or is_scored.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"Sequence {seq_ix}: expected [20000] masks")

        # Infer every step, including warm-up and unscored rows, before masking.
        predictions = predict_sequence(model, x, device, chunk_size)
        mask = need_prediction & is_scored
        # One add per complete sequence, never per chunk or per-sequence scores.
        accumulator.add(target.numpy(), predictions.numpy(), mask.numpy())
        print(
            f"  valid {row_group + 1}/{count} (seq_ix={seq_ix}): "
            f"selected_rows={int(mask.sum())}", flush=True,
        )

    report = accumulator.result()
    report["sequences"] = report["blocks"]
    return report


def main(argv=None):
    args = parse_args(argv)
    config, chunk_size, results_path = resolve_config(args)
    output_path = args.output_json or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_validation.json"
    )
    for source in (args.checkpoint, results_path, args.data_path):
        if source is not None and output_path.resolve() == source.resolve():
            raise ValueError("Output JSON must not overwrite an input file")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    model = build_model(argparse.Namespace(**config))
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print(f"Model: {config['model']} | Device: {device} | Chunk size: {chunk_size}")
    with pq.ParquetFile(args.data_path) as parquet_file:
        total_sequences = parquet_file.metadata.num_row_groups
        report = evaluate(model, parquet_file, device, chunk_size, args.max_sequences)

    report.update({
        "checkpoint": str(args.checkpoint),
        "results_json": str(results_path) if results_path is not None else None,
        "model_config": config,
        "data_path": str(args.data_path),
        "device": str(device),
        "chunk_size": chunk_size,
        "max_sequences": args.max_sequences,
        "total_validation_sequences": total_sequences,
        "full_validation": report["sequences"] == total_sequences,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")

    print(f"t0: {report['t0']:.8f}")
    print(f"t1: {report['t1']:.8f}")
    print(f"global weighted_pearson: {report['weighted_pearson']:.8f}")
    print(f"Sequences: {report['sequences']}")
    print(f"Selected rows: {report['selected_rows']}")
    print(f"Saved metric report: {output_path}")
    return report


if __name__ == "__main__":
    main()
