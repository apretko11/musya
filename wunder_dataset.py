import numpy as np
import pyarrow.parquet as pq
import torch

from pathlib import Path

DATASET_DIR = (
    Path.home()
    / "wnn_connectome_starterpack"
    / "datasets"
)

TRAIN_PATH = str(
    DATASET_DIR / "train.parquet"
)

VALID_PATH = str(
    DATASET_DIR / "valid.parquet"
)


def instrument_columns(instrument):
    """
    Return the 52 feature names for one instrument.

    22 p features
    22 v features
     4 dp features
     4 dv features
    """

    columns = []

    columns += [
        f"i{instrument}_p{i}"
        for i in range(22)
    ]

    columns += [
        f"i{instrument}_v{i}"
        for i in range(22)
    ]

    columns += [
        f"i{instrument}_dp{i}"
        for i in range(4)
    ]

    columns += [
        f"i{instrument}_dv{i}"
        for i in range(4)
    ]

    return columns


I0_COLUMNS = instrument_columns(0)
I1_COLUMNS = instrument_columns(1)

SHARED_COLUMNS = [
    f"a{i}"
    for i in range(8)
]

TARGET_COLUMNS = [
    "t0",
    "t1",
]

META_COLUMNS = [
    "seq_ix",
    "step_in_seq",
    "need_prediction",
]


def column_to_numpy(table, name):
    """
    Convert one PyArrow column into a NumPy array.
    """

    return table.column(name).to_numpy(
        zero_copy_only=False
    )


def load_sequence(
    parquet_file,
    row_group_index,
):
    """
    Load one Wunder sequence from one Parquet row group.

    Returns
    -------
    x:
        [T, 2, 60]

        T = timesteps
        2 = instruments
        60 = 52 instrument features
             + 8 shared features

    target:
        [T, 2]

        Last dimension is:
            target 0 = t0
            target 1 = t1

        IMPORTANT:
        This dimension is NOT the instrument dimension.

    mask:
        [T]

        True where a prediction is required.

    seq_ix:
        Sequence identifier.
    """

    columns = (
        META_COLUMNS
        + I0_COLUMNS
        + I1_COLUMNS
        + SHARED_COLUMNS
        + TARGET_COLUMNS
    )

    table = parquet_file.read_row_group(
        row_group_index,
        columns=columns,
    )

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    seq_ids = column_to_numpy(
        table,
        "seq_ix",
    )

    steps = column_to_numpy(
        table,
        "step_in_seq",
    )

    mask = column_to_numpy(
        table,
        "need_prediction",
    ).astype(bool)

    unique_sequences = np.unique(seq_ids)

    if len(unique_sequences) != 1:
        raise ValueError(
            "Expected one sequence per row group, "
            f"found {len(unique_sequences)}"
        )

    seq_ix = int(unique_sequences[0])

    # Verify ordering.
    expected_steps = np.arange(len(steps))

    if not np.array_equal(
        steps,
        expected_steps,
    ):
        raise ValueError(
            "step_in_seq is not contiguous from 0."
        )

    # -------------------------------------------------
    # Instrument-specific features
    # -------------------------------------------------

    i0 = np.column_stack([
        column_to_numpy(table, name)
        for name in I0_COLUMNS
    ]).astype(np.float32)

    i1 = np.column_stack([
        column_to_numpy(table, name)
        for name in I1_COLUMNS
    ]).astype(np.float32)

    # -------------------------------------------------
    # Shared features
    # -------------------------------------------------

    shared = np.column_stack([
        column_to_numpy(table, name)
        for name in SHARED_COLUMNS
    ]).astype(np.float32)

    # For the baseline, append the shared context
    # to both instruments.
    #
    # [T, 52] + [T, 8] -> [T, 60]

    i0 = np.concatenate(
        [i0, shared],
        axis=1,
    )

    i1 = np.concatenate(
        [i1, shared],
        axis=1,
    )

    # Put instruments on their own axis:
    #
    # [T, 2, 60]

    x = np.stack(
        [i0, i1],
        axis=1,
    )

    # -------------------------------------------------
    # Targets
    # -------------------------------------------------

    target = np.column_stack([
        column_to_numpy(table, name)
        for name in TARGET_COLUMNS
    ]).astype(np.float32)

    return (
        torch.from_numpy(x),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        seq_ix,
    )

def load_validation_sequence(
    parquet_file,
    row_group_index,
):
    """
    Load one complete labeled Wunder validation sequence.

    Returns
    -------
    x:
        [T, 2, 60]

    target:
        [T, 2]

    need_prediction:
        [T]

    is_scored:
        [T]

    seq_ix:
        sequence identifier
    """

    columns = (
        META_COLUMNS
        + ["is_scored"]
        + I0_COLUMNS
        + I1_COLUMNS
        + SHARED_COLUMNS
        + TARGET_COLUMNS
    )

    table = parquet_file.read_row_group(
        row_group_index,
        columns=columns,
    )

    seq_ids = column_to_numpy(
        table,
        "seq_ix",
    )

    steps = column_to_numpy(
        table,
        "step_in_seq",
    )

    need_prediction = column_to_numpy(
        table,
        "need_prediction",
    ).astype(bool)

    is_scored = column_to_numpy(
        table,
        "is_scored",
    ).astype(bool)

    unique_sequences = np.unique(
        seq_ids
    )

    if len(unique_sequences) != 1:
        raise ValueError(
            "Expected one sequence per row group, "
            f"found {len(unique_sequences)}"
        )

    seq_ix = int(
        unique_sequences[0]
    )

    expected_steps = np.arange(
        len(steps)
    )

    if not np.array_equal(
        steps,
        expected_steps,
    ):
        raise ValueError(
            "step_in_seq is not contiguous from 0."
        )

    i0 = np.column_stack([
        column_to_numpy(table, name)
        for name in I0_COLUMNS
    ]).astype(np.float32)

    i1 = np.column_stack([
        column_to_numpy(table, name)
        for name in I1_COLUMNS
    ]).astype(np.float32)

    shared = np.column_stack([
        column_to_numpy(table, name)
        for name in SHARED_COLUMNS
    ]).astype(np.float32)

    i0 = np.concatenate(
        [i0, shared],
        axis=1,
    )

    i1 = np.concatenate(
        [i1, shared],
        axis=1,
    )

    x = np.stack(
        [i0, i1],
        axis=1,
    )

    target = np.column_stack([
        column_to_numpy(table, name)
        for name in TARGET_COLUMNS
    ]).astype(np.float32)

    return (
        torch.from_numpy(x),
        torch.from_numpy(target),
        torch.from_numpy(need_prediction),
        torch.from_numpy(is_scored),
        seq_ix,
    )

if __name__ == "__main__":

    pf = pq.ParquetFile(
        TRAIN_PATH
    )

    print(
        "Number of row groups:",
        pf.metadata.num_row_groups,
    )

    x, target, mask, seq_ix = load_sequence(
        pf,
        row_group_index=0,
    )

    print()
    print("Sequence:", seq_ix)
    print("Input shape:", x.shape)
    print("Target shape:", target.shape)
    print("Mask shape:", mask.shape)

    print()
    print(
        "Prediction steps:",
        mask.sum().item(),
    )

    if mask.any():
        prediction_indices = torch.where(mask)[0]

        print(
            "First prediction step:",
            prediction_indices[0].item(),
        )

        print(
            "Last prediction step:",
            prediction_indices[-1].item(),
        )

    print()
    print(
        "Input NaNs:",
        torch.isnan(x).sum().item(),
    )

    print(
        "Target NaNs:",
        torch.isnan(target).sum().item(),
    )

    print()
    print(
        "First timestep, instrument 0:",
        x[0, 0, :10],
    )

    print(
        "First timestep, instrument 1:",
        x[0, 1, :10],
    )
