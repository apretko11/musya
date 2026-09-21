import numpy as np

from wunder_metric import (
    GlobalAccumulator,
)


rng = np.random.default_rng(0)

targets = rng.normal(
    size=(20000, 2)
).astype(np.float32)

mask = np.zeros(
    20000,
    dtype=bool,
)

mask[99:] = True


# Perfect predictions.
acc = GlobalAccumulator()

acc.add(
    targets,
    targets.copy(),
    mask,
)

print(
    "Perfect prediction:"
)

print(
    acc.result()
)


# Completely constant predictions.
constant = np.zeros_like(
    targets
)

acc = GlobalAccumulator()

acc.add(
    targets,
    constant,
    mask,
)

print()
print(
    "Constant prediction:"
)

print(
    acc.result()
)
