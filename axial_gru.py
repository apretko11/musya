import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Convert raw numerical + categorical features into a d_model vector.

    Input:
        numerical:   [B, T, A, num_numerical]
        categorical: [B, T, A, num_categorical]

    Output:
        x:           [B, T, A, d_model]
    """

    def __init__(
        self,
        num_numerical=77,
        categorical_cardinalities=(82, 13, 535),
        embedding_dim=16,
        d_model=64,
        clip_value=10.0,
    ):
        super().__init__()

        self.clip_value = clip_value

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities
        ])

        input_dim = (
            num_numerical
            + len(categorical_cardinalities) * embedding_dim
        )

        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, numerical, categorical):
        # Patrick clips extreme numerical values.
        numerical = torch.clamp(
            numerical,
            -self.clip_value,
            self.clip_value,
        )

        categorical_embeddings = []

        for i, embedding in enumerate(self.embeddings):
            e = embedding(categorical[..., i])
            categorical_embeddings.append(e)

        x = torch.cat(
            [numerical] + categorical_embeddings,
            dim=-1,
        )

        return self.projection(x)


class AssetMixer(nn.Module):
    """
    Self-attention across assets at each timestep.

    Input/output:
        [B, T, A, D]
    """

    def __init__(self, d_model=64, num_heads=4):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        B, T, A, D = x.shape

        residual = x

        x = self.norm(x)

        # Treat the A assets at each timestep as the
        # attention sequence.
        x = x.reshape(B * T, A, D)

        x, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        x = x.reshape(B, T, A, D)

        return residual + x


class TemporalGRU(nn.Module):
    """
    GRU across time independently for every asset.

    Input/output:
        [B, T, A, D]
    """

    def __init__(
        self,
        d_model=64,
        rnn_multiplier=4,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        hidden_dim = d_model * rnn_multiplier

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Project the larger GRU hidden state back down
        # to d_model.
        self.projection = nn.Linear(
            hidden_dim,
            d_model,
        )

    def forward(self, x, hidden=None):
        B, T, A, D = x.shape

        residual = x

        x = self.norm(x)

        # [B, T, A, D]
        #
        # We want one independent time sequence
        # for every asset.
        x = x.permute(0, 2, 1, 3)

        # [B, A, T, D]
        x = x.reshape(B * A, T, D)

        x, hidden = self.gru(x, hidden)

        x = self.projection(x)

        # Restore original axes.
        x = x.reshape(B, A, T, D)
        x = x.permute(0, 2, 1, 3)

        # [B, T, A, D]
        return residual + x, hidden


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model=64,
        expansion=4,
    ):
        super().__init__()

        hidden_dim = d_model * expansion

        self.norm = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class AxialGRUBlock(nn.Module):
    """
    Patrick-style block:

        Asset Mixer
            ->
        Temporal GRU
            ->
        FFN
    """

    def __init__(
        self,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        ffn_expansion=4,
    ):
        super().__init__()

        self.asset_mixer = AssetMixer(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.temporal_mixer = TemporalGRU(
            d_model=d_model,
            rnn_multiplier=rnn_multiplier,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            expansion=ffn_expansion,
        )

    def forward(self, x, hidden=None):
        x = self.asset_mixer(x)

        x, hidden = self.temporal_mixer(
            x,
            hidden,
        )

        x = self.ffn(x)

        return x, hidden


class AxialGRUModel(nn.Module):
    def __init__(
        self,
        num_numerical=77,
        categorical_cardinalities=(82, 13, 535),
        embedding_dim=16,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        num_blocks=3,
        num_targets=9,
    ):
        super().__init__()

        self.encoder = FeatureEncoder(
            num_numerical=num_numerical,
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=embedding_dim,
            d_model=d_model,
        )

        self.blocks = nn.ModuleList([
            AxialGRUBlock(
                d_model=d_model,
                num_heads=num_heads,
                rnn_multiplier=rnn_multiplier,
            )
            for _ in range(num_blocks)
        ])

        self.head = nn.Linear(
            d_model,
            num_targets,
        )

    def forward(self, numerical, categorical):
        x = self.encoder(
            numerical,
            categorical,
        )

        for block in self.blocks:
            x, _ = block(x)

        predictions = self.head(x)

        return predictions

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def print_parameter_breakdown(model):
    print("\nParameter breakdown:")

    for name, module in model.named_children():
        params = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        print(f"  {name:20s} {params:>10,}")

def make_synthetic_target(numerical):
    """
    Construct a synthetic prediction target that depends on:

    1. current features
    2. previous timestep
    3. cross-asset information

    Output:
        [B, T, A, 9]
    """

    current = numerical[..., :9]

    previous = torch.zeros_like(current)
    previous[:, 1:, :, :] = current[:, :-1, :, :]

    cross_asset = current.mean(
        dim=2,
        keepdim=True,
    )

    cross_asset = cross_asset.expand_as(current)

    target = (
        0.6 * current
        + 0.3 * previous
        + 0.1 * cross_asset
    )

    return target

def train_smoke_test(
    model,
    numerical,
    categorical,
    target,
    steps=50,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    model.train()

    for step in range(steps):
        optimizer.zero_grad()

        predictions = model(
            numerical,
            categorical,
        )

        loss = loss_fn(
            predictions,
            target,
        )

        loss.backward()

        optimizer.step()

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"  step {step + 1:3d}: "
                f"loss = {loss.item():.6f}"
            )

class LoopedAxialGRUModel(nn.Module):
    """
    Looped version of the Axial-GRU model.

    Instead of using N independently parameterized
    AxialGRUBlocks, we create ONE block and reuse it
    num_iterations times.

        H_0
         |
         v
      [Block] ----+
         |        |
         +--------+
         |
       repeat K times
         |
         v
      prediction
    """

    def __init__(
        self,
        num_numerical=77,
        categorical_cardinalities=(82, 13, 535),
        embedding_dim=16,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        num_iterations=3,
        num_targets=9,
    ):
        super().__init__()

        self.num_iterations = num_iterations

        self.encoder = FeatureEncoder(
            num_numerical=num_numerical,
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=embedding_dim,
            d_model=d_model,
        )

        # IMPORTANT:
        # There is only ONE AxialGRUBlock here.
        self.block = AxialGRUBlock(
            d_model=d_model,
            num_heads=num_heads,
            rnn_multiplier=rnn_multiplier,
        )

        self.head = nn.Linear(
            d_model,
            num_targets,
        )

    def forward(self, numerical, categorical):
        x = self.encoder(
            numerical,
            categorical,
        )

        # Same block, same weights, applied repeatedly.
        for _ in range(self.num_iterations):
            x, _ = self.block(x)

        predictions = self.head(x)

        return predictions

class WunderFeatureEncoder(nn.Module):
    """
    Wunder input:

        [B, T, A, 60]

    Output:

        [B, T, A, d_model]
    """

    def __init__(
        self,
        input_dim=60,
        d_model=64,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            d_model,
        )

    def forward(self, x):
        return self.projection(x)


class WunderAxialGRUModel(nn.Module):
    """
    Patrick-style baseline for Wunder.

    Uses N independently parameterized axial blocks.
    """

    def __init__(
        self,
        input_dim=60,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        num_blocks=3,
        num_targets=2,
    ):
        super().__init__()

        self.encoder = WunderFeatureEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        self.blocks = nn.ModuleList([
            AxialGRUBlock(
                d_model=d_model,
                num_heads=num_heads,
                rnn_multiplier=rnn_multiplier,
            )
            for _ in range(num_blocks)
        ])

        self.head = nn.Linear(
            d_model,
            num_targets,
        )

    def forward(
        self,
        x,
        hidden_states=None,
        return_hidden=False,
    ):
        x = self.encoder(x)

        if hidden_states is None:
            hidden_states = [
                None
                for _ in self.blocks
            ]

        new_hidden_states = []

        for block, hidden in zip(
            self.blocks,
            hidden_states,
        ):
            x, new_hidden = block(
                x,
                hidden,
            )

            new_hidden_states.append(
                new_hidden
            )

        target_asset = x[:, :, 0, :]

        predictions = self.head(
            target_asset
        )

        if return_hidden:
            return (
                predictions,
                new_hidden_states,
            )

        return predictions


class WunderLoopedAxialGRUModel(nn.Module):
    """
    Weight-tied version.

    One axial block is reused num_iterations times.
    """

    def __init__(
        self,
        input_dim=60,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        num_iterations=3,
        num_targets=2,
    ):
        super().__init__()

        self.num_iterations = num_iterations

        self.encoder = WunderFeatureEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        self.block = AxialGRUBlock(
            d_model=d_model,
            num_heads=num_heads,
            rnn_multiplier=rnn_multiplier,
        )

        self.head = nn.Linear(
            d_model,
            num_targets,
        )

    def forward(
        self,
        x,
        hidden_states=None,
        return_hidden=False,
    ):
        x = self.encoder(x)

        if hidden_states is None:
            hidden_states = [
                None
                for _ in range(
                    self.num_iterations
                )
            ]

        new_hidden_states = []

        for iteration in range(
            self.num_iterations
        ):
            x, new_hidden = self.block(
                x,
                hidden_states[iteration],
            )

            new_hidden_states.append(
                new_hidden
            )

        target_asset = x[:, :, 0, :]

        predictions = self.head(
            target_asset
        )

        if return_hidden:
            return (
                predictions,
                new_hidden_states,
            )

        return predictions

class WunderReinjectedLoopedAxialGRUModel(nn.Module):
    """
    Looped Axial-GRU with input reinjection.

    The same AxialGRUBlock is reused at every
    refinement iteration.

    Before iterations 2...K, the current hidden
    representation is mixed with the original
    encoded input.

        H0 = Encoder(X)

        H_{k+1} = Block(
            (1 - g) * H_k + g * H0
        )

    where g is learned.
    """

    def __init__(
        self,
        input_dim=60,
        d_model=64,
        num_heads=4,
        rnn_multiplier=4,
        num_iterations=3,
        num_targets=2,
    ):
        super().__init__()

        self.num_iterations = num_iterations

        self.encoder = WunderFeatureEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        self.block = AxialGRUBlock(
            d_model=d_model,
            num_heads=num_heads,
            rnn_multiplier=rnn_multiplier,
        )

        # Start with relatively weak reinjection.
        #
        # sigmoid(-2) ~= 0.12
        self.reinjection_logit = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.head = nn.Linear(
            d_model,
            num_targets,
        )

    def forward(
        self,
        x,
        hidden_states=None,
        return_hidden=False,
    ):
        # Original encoded observation.
        h0 = self.encoder(x)

        x = h0

        if hidden_states is None:
            hidden_states = [
                None
                for _ in range(
                    self.num_iterations
                )
            ]

        new_hidden_states = []

        # Shared scalar gate in [0, 1].
        gate = torch.sigmoid(
            self.reinjection_logit
        )

        for iteration in range(
            self.num_iterations
        ):

            # First iteration already receives h0,
            # so reinjection starts on iteration 2.
            if iteration > 0:
                x = (
                    (1.0 - gate) * x
                    + gate * h0
                )

            x, new_hidden = self.block(
                x,
                hidden_states[iteration],
            )

            new_hidden_states.append(
                new_hidden
            )

        target_asset = x[:, :, 0, :]

        predictions = self.head(
            target_asset
        )

        if return_hidden:
            return (
                predictions,
                new_hidden_states,
            )

        return predictions

if __name__ == "__main__":

    B = 2
    T = 10
    A = 5

    numerical = torch.randn(
        B,
        T,
        A,
        77,
    )

    feature_09 = torch.randint(
        0,
        82,
        (B, T, A, 1),
    )

    feature_10 = torch.randint(
        0,
        13,
        (B, T, A, 1),
    )

    feature_11 = torch.randint(
        0,
        535,
        (B, T, A, 1),
    )

    categorical = torch.cat(
        [
            feature_09,
            feature_10,
            feature_11,
        ],
        dim=-1,
    )

    baseline = AxialGRUModel(
        num_blocks=3,
    )

    looped = LoopedAxialGRUModel(
        num_iterations=3,
    )

    baseline_predictions = baseline(
        numerical,
        categorical,
    )

    looped_predictions = looped(
        numerical,
        categorical,
    )

    print("Input:")
    print("  numerical:", numerical.shape)
    print("  categorical:", categorical.shape)

    print()

    print("Baseline:")
    print(
        "  predictions:",
        baseline_predictions.shape,
    )
    print(
        "  parameters:",
        count_parameters(baseline),
    )

    print()

    print("Looped:")
    print(
        "  predictions:",
        looped_predictions.shape,
    )
    print(
        "  parameters:",
        count_parameters(looped),
    )

    print("\nBaseline breakdown:")
    print_parameter_breakdown(baseline)

    print("\nLooped breakdown:")
    print_parameter_breakdown(looped)
    
    target = make_synthetic_target(
        numerical,
    )

    print("\nSynthetic target:")
    print("  target:", target.shape)

    print("\nTraining baseline:")
    train_smoke_test(
        baseline,
        numerical,
        categorical,
        target,
    )

    print("\nTraining looped model:")
    train_smoke_test(
        looped,
        numerical,
        categorical,
        target,
    )

