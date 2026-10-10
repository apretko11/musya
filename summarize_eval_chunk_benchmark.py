import json
from pathlib import Path


directory = Path(
    "results/eval_chunk_benchmark"
)

print(
    f"{'Chunk':>8} "
    f"{'WP':>12} "
    f"{'t0':>12} "
    f"{'t1':>12} "
    f"{'Seconds':>12}"
)

print("-" * 60)

for path in sorted(
    directory.glob("chunk_*.json"),
    key=lambda p: int(
        p.stem.split("_")[1]
    ),
):
    with path.open() as f:
        data = json.load(f)

    print(
        f"{data['chunk_size']:8d} "
        f"{data['weighted_pearson']:12.8f} "
        f"{data['t0']:12.8f} "
        f"{data['t1']:12.8f} "
        f"{data['elapsed_seconds']:12.2f}"
    )
