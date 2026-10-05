# `wunder_dataset.py`

## Purpose

`wunder_dataset.py` is a small data-loading utility for the Wunder Connectome dataset.

Its job is to:

1. Locate the training and validation Parquet files.
2. Define the dataset column groups.
3. Load one complete sequence from one Parquet row group.
4. Rearrange the raw columns into model-friendly NumPy arrays.
5. Convert those arrays into PyTorch tensors.
6. Perform basic sanity checks on the dataset structure.

The resulting model input has shape:

```text
[T, 2, 60]
```

where:

- `T` = number of timesteps in the sequence.
- `2` = the two instruments, `i0` and `i1`.
- `60` = 52 instrument-specific features + 8 shared features.

---

## Dataset paths

```python
DATASET_DIR = (
    Path.home()
    / "wnn_connectome_starterpack"
    / "datasets"
)
```

`DATASET_DIR` is a `Path` object pointing to:

```text
~/wnn_connectome_starterpack/datasets/
```

The actual file paths are then created:

```python
TRAIN_PATH = str(
    DATASET_DIR / "train.parquet"
)

VALID_PATH = str(
    DATASET_DIR / "valid.parquet"
)
```

`DATASET_DIR / "train.parquet"` initially creates another `Path` object.

`str(...)` converts it into a normal Python string before it is passed to PyArrow.

---

# Dataset column structure

## Instrument-specific features

The helper:

```python
instrument_columns(instrument)
```

generates the 52 feature names belonging to one instrument.

Each instrument contains:

```text
22 p features
22 v features
 4 dp features
 4 dv features
--------------
52 features
```

For example:

```python
I0_COLUMNS = instrument_columns(0)
```

produces names such as:

```text
i0_p0
i0_p1
...
i0_p21

i0_v0
...
i0_v21

i0_dp0
...
i0_dp3

i0_dv0
...
i0_dv3
```

Likewise:

```python
I1_COLUMNS = instrument_columns(1)
```

contains the corresponding 52 columns for instrument 1.

These lists contain **column names**, not the actual feature values.

---

## Shared features

```python
SHARED_COLUMNS = [
    "a0",
    "a1",
    ...
    "a7",
]
```

There are 8 additional features that are not specific to either instrument.

The loader later appends these same 8 shared features to both instruments.

Thus each instrument ultimately gets:

```text
52 instrument-specific
+ 8 shared
----------------------
60 features
```

---

## Targets

```python
TARGET_COLUMNS = [
    "t0",
    "t1",
]
```

There are two prediction targets.

The target tensor has shape:

```text
[T, 2]
```

Importantly, this dimension of size `2` represents:

```text
target 0 = t0
target 1 = t1
```

It does **not** represent the two instruments.

Both targets relate to instrument `i0`.

---

## Metadata

```python
META_COLUMNS = [
    "seq_ix",
    "step_in_seq",
    "need_prediction",
]
```

### `seq_ix`

Identifies which sequence a row belongs to.

All rows within a row group are expected to have the same `seq_ix`.

### `step_in_seq`

Identifies the timestep within the sequence.

The loader verifies that these are exactly:

```text
0, 1, 2, ..., T-1
```

with no missing or reordered timesteps.

### `need_prediction`

Boolean value indicating whether a prediction is required at that timestep.

This becomes a boolean mask of shape:

```text
[T]
```

---

# `column_to_numpy()`

```python
def column_to_numpy(table, name):
    return table.column(name).to_numpy(
        zero_copy_only=False
    )
```

This helper takes one column from a PyArrow table and converts it into a NumPy array.

For example:

```python
column_to_numpy(table, "i0_p0")
```

performs approximately:

```text
PyArrow column
      ↓
NumPy array of length T
```

`zero_copy_only=False` means PyArrow may copy the data if a zero-copy conversion is not possible.

---

# `load_sequence()`

```python
load_sequence(
    parquet_file,
    row_group_index,
)
```

Loads one complete training sequence.

The code assumes:

```text
1 Parquet row group = 1 Wunder sequence
```

---

## 1. Choose required columns

The function creates one list containing:

```text
metadata
+ instrument 0 features
+ instrument 1 features
+ shared features
+ targets
```

Then:

```python
table = parquet_file.read_row_group(
    row_group_index,
    columns=columns,
)
```

reads only those columns from the requested row group.

`read_row_group()` is a built-in method of PyArrow's `ParquetFile` class.

The result is a PyArrow `Table`.

---

## 2. Extract metadata

The function extracts:

```python
seq_ids
steps
mask
```

where:

```text
seq_ids = seq_ix for every timestep

steps = step_in_seq for every timestep

mask = need_prediction for every timestep
```

The mask is explicitly converted to boolean.

---

## 3. Validate the sequence

The loader checks two assumptions.

### Exactly one sequence

```python
unique_sequences = np.unique(seq_ids)
```

There must be exactly one unique sequence ID within the row group.

Otherwise a `ValueError` is raised.

After verification:

```python
seq_ix = int(unique_sequences[0])
```

stores the sequence ID as a regular Python integer.

### Correct timestep ordering

```python
expected_steps = np.arange(len(steps))
```

creates:

```text
0, 1, 2, ..., T-1
```

The loader checks that the actual `step_in_seq` values match this exactly.

This protects against missing or reordered timesteps.

---

# Constructing the model input

## Instrument 0

Each of the 52 instrument-0 columns is extracted:

```python
column_to_numpy(table, name)
```

giving 52 vectors, each with shape:

```text
[T]
```

Then:

```python
np.column_stack(...)
```

places the 52 vectors side by side:

```text
i0 → [T, 52]
```

The values are converted to:

```text
float32
```

---

## Instrument 1

The same process gives:

```text
i1 → [T, 52]
```

---

## Shared features

The 8 shared columns are stacked:

```text
shared → [T, 8]
```

---

## Append shared features

The shared values are appended to both instruments:

```python
i0 = np.concatenate(
    [i0, shared],
    axis=1,
)

i1 = np.concatenate(
    [i1, shared],
    axis=1,
)
```

Therefore:

```text
i0: [T,52] + [T,8]
             ↓
          [T,60]

i1: [T,52] + [T,8]
             ↓
          [T,60]
```

`axis=1` means concatenation occurs along the feature dimension.

---

## Add an instrument dimension

```python
x = np.stack(
    [i0, i1],
    axis=1,
)
```

`np.stack()` creates a new dimension.

Thus:

```text
i0 [T,60]
i1 [T,60]
     ↓
x [T,2,60]
```

The dimensions mean:

```text
x[timestep, instrument, feature]
```

For example:

```python
x[5, 0, :]
```

means:

> all 60 features of instrument 0 at timestep 5.

---

# Targets

```python
target = np.column_stack([
    column_to_numpy(table, name)
    for name in TARGET_COLUMNS
]).astype(np.float32)
```

The two target columns:

```text
t0
t1
```

are stacked together:

```text
target → [T,2]
```

with:

```text
target[t,0] = t0 at timestep t
target[t,1] = t1 at timestep t
```

---

# Conversion to PyTorch

`load_sequence()` returns:

```python
return (
    torch.from_numpy(x),
    torch.from_numpy(target),
    torch.from_numpy(mask),
    seq_ix,
)
```

Therefore:

```text
x
    torch.float32
    [T,2,60]

target
    torch.float32
    [T,2]

mask
    torch.bool
    [T]

seq_ix
    Python int
```

`torch.from_numpy()` converts the NumPy arrays into PyTorch tensors, generally without copying the underlying memory.

---

# `load_validation_sequence()`

This function is almost identical to `load_sequence()`.

The main additional field is:

```python
"is_scored"
```

It therefore returns:

```text
x
target
need_prediction
is_scored
seq_ix
```

instead of:

```text
x
target
mask
seq_ix
```

The distinction is:

```text
need_prediction
    → should the model produce a prediction here?

is_scored
    → does this timestep count toward the validation score?
```

Everything else — sequence validation, feature construction, shared-feature concatenation, stacking, and target creation — is essentially the same.

---

# `main`

```python
if __name__ == "__main__":
```

runs only when the file itself is executed:

```bash
python wunder_dataset.py
```

It does not run when the file is imported from another Python script.

The `main` block is essentially a smoke test for the loader.

---

## Open the training dataset

```python
pf = pq.ParquetFile(
    TRAIN_PATH
)
```

This creates a PyArrow `ParquetFile` for:

```text
train.parquet
```

---

## Count row groups

```python
pf.metadata.num_row_groups
```

prints the number of row groups.

Since this code assumes:

```text
1 row group = 1 sequence
```

this also effectively tells us the number of training sequences.

---

## Load the first training sequence

```python
x, target, mask, seq_ix = load_sequence(
    pf,
    row_group_index=0,
)
```

This loads row group `0`, i.e. the first training sequence.

The validation loader is **not used by `main`**.

---

## Print tensor shapes

The script prints:

```text
sequence ID
input shape
target shape
mask shape
```

This verifies that the loader produced the expected tensor dimensions.

---

## Count prediction timesteps

```python
mask.sum().item()
```

counts how many entries of `need_prediction` are `True`.

For boolean tensors:

```text
False = 0
True  = 1
```

so summing the mask counts the `True` values.

`.item()` extracts the scalar value from a 0-dimensional PyTorch tensor.

---

## Locate prediction region

```python
prediction_indices = torch.where(mask)[0]
```

finds all timestep indices where:

```text
need_prediction == True
```

The script then prints:

```text
first prediction timestep
last prediction timestep
```

as a sanity check.

---

## Check for NaNs

```python
torch.isnan(x).sum().item()
```

counts NaN values in the model inputs.

Likewise:

```python
torch.isnan(target).sum().item()
```

checks the targets.

Ideally both should return:

```text
0
```

---

## Inspect actual feature values

Finally:

```python
x[0, 0, :10]
```

prints:

> the first 10 features of instrument 0 at timestep 0.

And:

```python
x[0, 1, :10]
```

prints:

> the first 10 features of instrument 1 at timestep 0.

This is simply a quick visual sanity check.

---

# Overall data flow

```text
train.parquet / valid.parquet
            │
            ▼
     PyArrow ParquetFile
            │
            ▼
       one row group
            │
            │
      one sequence
            │
            ▼
   ┌───────────────────┐
   │ Metadata          │
   │ seq_ix            │
   │ step_in_seq       │
   │ need_prediction   │
   └───────────────────┘

i0 columns       i1 columns       shared
52 features      52 features      8 features
     │                │               │
     ▼                ▼               ▼
  [T,52]           [T,52]          [T,8]
     │                │
     └── append shared ──┐
                         │
          [T,60]      [T,60]
              \        /
               \      /
                stack
                  │
                  ▼
              x [T,2,60]

t0 + t1
   │
   ▼
target [T,2]

need_prediction
   │
   ▼
mask [T]
```

For validation, an additional:

```text
is_scored [T]
```

is also returned.

---

# In one sentence

`wunder_dataset.py` converts one raw Wunder Parquet sequence into validated, model-ready PyTorch tensors of inputs `[T,2,60]`, targets `[T,2]`, and prediction masks `[T]`, with an additional scoring mask for validation data.
