import pyarrow.parquet as pq

from wunder_dataset import (
    VALID_PATH,
    load_validation_sequence,
)


pf = pq.ParquetFile(
    VALID_PATH
)

x, target, need, scored, seq_ix = (
    load_validation_sequence(
        pf,
        row_group_index=0,
    )
)

print("Sequence:", seq_ix)
print("Input:", x.shape)
print("Target:", target.shape)
print("Need prediction:", need.shape)
print("Is scored:", scored.shape)

print()
print(
    "Prediction rows:",
    need.sum().item(),
)

print(
    "Scored rows:",
    scored.sum().item(),
)

print(
    "Official metric rows:",
    (need & scored).sum().item(),
)
