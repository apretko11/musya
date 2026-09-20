import pyarrow.parquet as pq
import torch
import torch.nn as nn

from axial_gru import (
    WunderAxialGRUModel,
    WunderLoopedAxialGRUModel,
)

from wunder_dataset import (
    TRAIN_PATH,
    load_sequence,
)


def train_model(
    model,
    x,
    target,
    mask,
    steps=20,
    lr=1e-3,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    loss_fn = nn.MSELoss()

    model.train()

    for step in range(steps):

        optimizer.zero_grad()

        prediction = model(x)

        # prediction: [B, T, 2]
        # target:     [B, T, 2]
        # mask:       [B, T]
        #
        # mask selects only timesteps where
        # need_prediction == True.

        loss = loss_fn(
            prediction[mask],
            target[mask],
        )

        loss.backward()

        optimizer.step()

        print(
            f"step {step + 1:2d}: "
            f"loss = {loss.item():.6f}"
        )


# --------------------------------------------------
# Load one real Wunder sequence
# --------------------------------------------------

pf = pq.ParquetFile(
    TRAIN_PATH
)

x, target, mask, seq_ix = load_sequence(
    pf,
    row_group_index=0,
)


# Keep this small for laptop testing.
T = 512

x = x[:T]
target = target[:T]
mask = mask[:T]


# Add batch dimension:
#
# [T, 2, 60] -> [1, T, 2, 60]
# [T, 2]     -> [1, T, 2]
# [T]        -> [1, T]

x = x.unsqueeze(0)
target = target.unsqueeze(0)
mask = mask.unsqueeze(0)


print("Sequence:", seq_ix)

print("Input:", x.shape)
print("Target:", target.shape)
print("Mask:", mask.shape)

print(
    "Prediction positions:",
    mask.sum().item(),
)


# --------------------------------------------------
# Baseline
# --------------------------------------------------

torch.manual_seed(0)

baseline = WunderAxialGRUModel(
    num_blocks=3,
)

print("\nTraining baseline:")

train_model(
    baseline,
    x,
    target,
    mask,
)


# --------------------------------------------------
# Looped model
# --------------------------------------------------

torch.manual_seed(0)

looped = WunderLoopedAxialGRUModel(
    num_iterations=3,
)

print("\nTraining looped model:")

train_model(
    looped,
    x,
    target,
    mask,
)
