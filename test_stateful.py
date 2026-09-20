import pyarrow.parquet as pq
import torch

from axial_gru import (
    WunderAxialGRUModel,
    WunderLoopedAxialGRUModel,
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

# Use 512 steps.
x = x[:512].unsqueeze(0)


def test_model(model, name):

    model.eval()

    with torch.no_grad():

        # --------------------------------
        # Full 512-step computation
        # --------------------------------

        full = model(x)

        # --------------------------------
        # Chunked computation
        # --------------------------------

        hidden = None
        outputs = []

        for start in range(
            0,
            512,
            128,
        ):
            chunk = x[
                :,
                start:start + 128,
            ]

            prediction, hidden = model(
                chunk,
                hidden_states=hidden,
                return_hidden=True,
            )

            outputs.append(
                prediction
            )

        chunked = torch.cat(
            outputs,
            dim=1,
        )

        difference = (
            full - chunked
        ).abs()

        print(name)

        print(
            "  full:",
            full.shape,
        )

        print(
            "  chunked:",
            chunked.shape,
        )

        print(
            "  max difference:",
            difference.max().item(),
        )

        print(
            "  mean difference:",
            difference.mean().item(),
        )

        print()


torch.manual_seed(0)

baseline = WunderAxialGRUModel(
    num_blocks=3,
)

test_model(
    baseline,
    "Baseline",
)


torch.manual_seed(0)

looped = WunderLoopedAxialGRUModel(
    num_iterations=3,
)

test_model(
    looped,
    "Looped",
)

torch.manual_seed(0)

reinjected = WunderReinjectedLoopedAxialGRUModel(
    num_iterations=3,
)

test_model(
    reinjected,
    "Reinjected looped",
)
