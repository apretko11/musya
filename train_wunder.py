import pyarrow.parquet as pq
import torch
import torch.nn as nn

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


DEVICE = torch.device("cpu")


def detach_hidden_states(hidden_states):
    """
    Detach GRU hidden states from the previous
    truncated-BPTT computation graph.
    """

    if hidden_states is None:
        return None

    return [
        hidden.detach()
        for hidden in hidden_states
    ]


def train_sequence(
    model,
    optimizer,
    x,
    target,
    mask,
    chunk_size=512,
):
    """
    Train on one complete Wunder sequence using
    truncated backpropagation through time.
    """

    model.train()

    hidden_states = None

    total_loss = 0.0
    total_prediction_steps = 0

    T = x.shape[0]

    for start in range(
        0,
        T,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            T,
        )

        x_chunk = x[
            start:end
        ].unsqueeze(0).to(DEVICE)

        target_chunk = target[
            start:end
        ].unsqueeze(0).to(DEVICE)

        mask_chunk = mask[
            start:end
        ].unsqueeze(0).to(DEVICE)

        # Some chunks may theoretically contain no
        # prediction positions.
        num_prediction_steps = (
            mask_chunk.sum().item()
        )

        optimizer.zero_grad()

        prediction, hidden_states = model(
            x_chunk,
            hidden_states=hidden_states,
            return_hidden=True,
        )

        # Detach before moving to the next chunk.
        #
        # This preserves the GRU state values while
        # truncating the gradient graph.
        hidden_states = detach_hidden_states(
            hidden_states
        )

        if num_prediction_steps == 0:
            continue

        loss = nn.functional.mse_loss(
            prediction[mask_chunk],
            target_chunk[mask_chunk],
        )

        loss.backward()

        # GRUs + recurrent-depth models can produce
        # large gradients, so clip them.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * num_prediction_steps
        )

        total_prediction_steps += (
            num_prediction_steps
        )

    return (
        total_loss
        / total_prediction_steps
    )


def train_model(
    model,
    parquet_file,
    row_groups,
    epochs=2,
    chunk_size=512,
    lr=1e-4,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print(
        "Parameters:",
        count_parameters(model),
    )

    for epoch in range(epochs):

        epoch_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        for sequence_number, row_group in enumerate(
            row_groups,
            start=1,
        ):
            x, target, mask, seq_ix = load_sequence(
                parquet_file,
                row_group,
            )

            loss = train_sequence(
                model,
                optimizer,
                x,
                target,
                mask,
                chunk_size=chunk_size,
            )

            epoch_loss += loss

            print(
                f"  sequence "
                f"{sequence_number:2d}/"
                f"{len(row_groups)} "
                f"(seq_ix={seq_ix}): "
                f"loss = {loss:.6f}"
            )

        mean_loss = (
            epoch_loss
            / len(row_groups)
        )

        print(
            f"Epoch mean loss: "
            f"{mean_loss:.6f}"
        )

def evaluate_sequence(
    model,
    x,
    target,
    mask,
    chunk_size=512,
):
    model.eval()

    hidden_states = None

    total_loss = 0.0
    total_prediction_steps = 0

    T = x.shape[0]

    with torch.no_grad():

        for start in range(
            0,
            T,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                T,
            )

            x_chunk = x[
                start:end
            ].unsqueeze(0).to(DEVICE)

            target_chunk = target[
                start:end
            ].unsqueeze(0).to(DEVICE)

            mask_chunk = mask[
                start:end
            ].unsqueeze(0).to(DEVICE)

            prediction, hidden_states = model(
                x_chunk,
                hidden_states=hidden_states,
                return_hidden=True,
            )

            num_prediction_steps = (
                mask_chunk.sum().item()
            )

            if num_prediction_steps == 0:
                continue

            loss = nn.functional.mse_loss(
                prediction[mask_chunk],
                target_chunk[mask_chunk],
            )

            total_loss += (
                loss.item()
                * num_prediction_steps
            )

            total_prediction_steps += (
                num_prediction_steps
            )

    return (
        total_loss
        / total_prediction_steps
    )

def evaluate_model(
    model,
    parquet_file,
    row_groups,
    chunk_size=512,
):
    print("\nValidation:")

    total_loss = 0.0

    for sequence_number, row_group in enumerate(
        row_groups,
        start=1,
    ):
        x, target, mask, seq_ix = load_sequence(
            parquet_file,
            row_group,
        )

        loss = evaluate_sequence(
            model,
            x,
            target,
            mask,
            chunk_size=chunk_size,
        )

        total_loss += loss

        print(
            f"  sequence "
            f"{sequence_number:2d}/"
            f"{len(row_groups)} "
            f"(seq_ix={seq_ix}): "
            f"loss = {loss:.6f}"
        )

    mean_loss = (
        total_loss
        / len(row_groups)
    )

    print(
        f"Validation mean loss: "
        f"{mean_loss:.6f}"
    )

    return mean_loss

if __name__ == "__main__":

    torch.manual_seed(0)

    pf = pq.ParquetFile(
        TRAIN_PATH
    )

    # Keep this deliberately tiny for the laptop.
    #
    # Four complete sequences:
    # 4 × 20,000 = 80,000 timesteps.
    train_row_groups = [
        0,
        1,
        2,
        3,
    ]
    
    
    valid_row_groups = [
        4,
        5,
    ]

    print(
        "Training row groups:",
        train_row_groups,
    )

    # --------------------------------------------
    # For now train ONLY ONE model at a time.
    # Start with Patrick-style baseline.
    # --------------------------------------------

    model = WunderReinjectedLoopedAxialGRUModel(
        num_iterations=3,
    ).to(DEVICE)

    train_model(
        model,
        pf,
        train_row_groups,
        epochs=2,
        chunk_size=512,
        lr=1e-4,
    )
    
    evaluate_model(
        model,
        pf,
        valid_row_groups,
        chunk_size=512,
    )
    
    if hasattr(model, "reinjection_logit"):
        print(
            "\nLearned reinjection gate:",
            torch.sigmoid(
                model.reinjection_logit
            ).item()
        )

    torch.save(
        model.state_dict(),
        "reinjected_model.pt",
    )
