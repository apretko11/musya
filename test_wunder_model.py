import pyarrow.parquet as pq
import torch

from axial_gru import (
    WunderAxialGRUModel,
    WunderLoopedAxialGRUModel,
    count_parameters,
    WunderReinjectedLoopedAxialGRUModel,
)

from wunder_dataset import (
    TRAIN_PATH,
    load_sequence,
)


pf = pq.ParquetFile(
    TRAIN_PATH
)

x, target, mask, seq_ix = load_sequence(
    pf,
    row_group_index=0,
)

# Use only a small chunk on the laptop.
T = 512

x = x[:T]
target = target[:T]
mask = mask[:T]

# Add batch dimension:
#
# [T, 2, 60]
# ->
# [1, T, 2, 60]

x = x.unsqueeze(0)

target = target.unsqueeze(0)
mask = mask.unsqueeze(0)


baseline = WunderAxialGRUModel(
    num_blocks=3,
)

looped = WunderLoopedAxialGRUModel(
    num_iterations=3,
)


with torch.no_grad():

    baseline_pred = baseline(x)

    looped_pred = looped(x)


print("Sequence:", seq_ix)

print()

print("Input:")
print(" ", x.shape)

print("Target:")
print(" ", target.shape)

print("Mask:")
print(" ", mask.shape)

print()

print("Baseline:")
print("  output:", baseline_pred.shape)
print(
    "  parameters:",
    count_parameters(baseline),
)

print()

print("Looped:")
print("  output:", looped_pred.shape)
print(
    "  parameters:",
    count_parameters(looped),
)

print()

print(
    "Prediction positions in chunk:",
    mask.sum().item(),
)

##########REINJECTED#######

reinjected = WunderReinjectedLoopedAxialGRUModel(
    num_iterations=3,
)

with torch.no_grad():
    reinjected_pred = reinjected(x)

print()

print("Reinjected looped:")
print(
    "  output:",
    reinjected_pred.shape,
)

print(
    "  parameters:",
    count_parameters(reinjected),
)

print(
    "  gate:",
    torch.sigmoid(
        reinjected.reinjection_logit
    ).item(),
)
