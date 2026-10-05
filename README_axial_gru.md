# `axial_gru.py`

## Purpose

`axial_gru.py` defines the neural-network architectures used for the Wunder experiments.

The file has two main jobs:

1. Define the reusable components of one Axial-GRU block:
   - `AssetMixer`
   - `TemporalGRU`
   - `FeedForward`
   - `AxialGRUBlock`

2. Define the three model variants being compared:
   - `WunderAxialGRUModel`
   - `WunderLoopedAxialGRUModel`
   - `WunderReinjectedLoopedAxialGRUModel`

The basic idea is to process the Wunder data along two different structural dimensions:

```text
asset / instrument dimension
        i0 ↔ i1

time dimension
        t0 → t1 → t2 → ...
```

The asset dimension is modeled using self-attention.

The time dimension is modeled using a GRU.

One complete Axial-GRU block is therefore:

```text
AssetMixer
    ↓
TemporalGRU
    ↓
FeedForward
```

---

# Tensor notation

Throughout the file, tensors generally use the shape:

```text
[B, T, A, D]
```

where:

```text
B = batch size
T = number of timesteps
A = number of assets / instruments
D = latent model dimension
```

For Wunder:

```text
A = 2
```

because there are two instruments:

```text
i0
i1
```

The raw feature dimension is:

```text
60
```

per instrument after preprocessing.

The default latent model dimension is:

```text
D = 64
```

Thus a typical training chunk has shape:

```text
raw input:
[1, 512, 2, 60]

after feature encoding:
[1, 512, 2, 64]
```

The complete Wunder sequence contains 20,000 timesteps, but training and evaluation typically process it in smaller chunks while preserving GRU state between chunks.

---

# `AssetMixer`

```python
class AssetMixer(nn.Module):
```

`AssetMixer` performs self-attention across the instruments at each timestep.

It does not process temporal relationships.

For a tensor:

```text
[B, T, A, D]
```

the conceptual operation is:

```text
time 0:  i0 ↔ i1
time 1:  i0 ↔ i1
time 2:  i0 ↔ i1
...
```

Each timestep is treated as an independent attention problem.

## Initialization

```python
def __init__(
    self,
    d_model=64,
    num_heads=4,
):
```

The defaults are:

```text
d_model = 64
num_heads = 4
```

Thus every instrument is represented by a 64-dimensional latent vector.

Multi-head attention divides this representation across four heads:

```text
64 / 4 = 16 dimensions per head
```

## Layer normalization

```python
self.norm = nn.LayerNorm(d_model)
```

Layer normalization operates on each 64-dimensional representation independently.

It does not mix timesteps, assets, or batch elements.

The tensor shape remains:

```text
[B, T, A, D]
```

## Multi-head attention

```python
self.attention = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True,
)
```

`batch_first=True` means PyTorch expects attention input in the form:

```text
[number of examples, sequence length, embedding dimension]
```

The attention tokens in this model are the instruments.

For Wunder:

```text
token 0 = i0
token 1 = i1
```

## `AssetMixer.forward()`

```python
B, T, A, D = x.shape
```

This extracts the tensor dimensions.

For a normal chunk:

```text
B = 1
T = 512
A = 2
D = 64
```

### Preserve the residual

```python
residual = x
```

The original representation is saved so that it can later be added back to the attention output.

The final operation is approximately:

```text
new representation
=
old representation
+
attention update
```

### Normalize

```python
x = self.norm(x)
```

Shape remains:

```text
[B, T, A, D]
```

### Reshape for asset attention

```python
x = x.reshape(B * T, A, D)
```

This is the operation that makes attention work across assets.

For:

```text
[1, 512, 2, 64]
```

the reshape gives:

```text
[512, 2, 64]
```

This means:

```text
512 independent attention examples
2 tokens per example
64 dimensions per token
```

Each example corresponds to one timestep:

```text
example 0:
    i0 at timestep 0
    i1 at timestep 0

example 1:
    i0 at timestep 1
    i1 at timestep 1

...
```

The 512 examples do not attend to one another.

Attention operates only over the `A=2` token dimension.

### Self-attention

```python
x, _ = self.attention(
    x,
    x,
    x,
    need_weights=False,
)
```

`nn.MultiheadAttention` expects:

```text
query
key
value
```

Using:

```python
x, x, x
```

means this is self-attention.

The same set of instrument tokens provides the queries, keys, and values.

Internally:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

For one attention head with two tokens:

```text
Q: [2, 16]
K: [2, 16]
V: [2, 16]
```

The attention-score matrix is:

```text
Q K^T
```

with shape:

```text
[2,16] × [16,2]
    ↓
[2,2]
```

Conceptually:

```text
                   key
                i0      i1

query i0      i0→i0   i0→i1
      i1      i1→i0   i1→i1
```

After scaling and softmax, these scores determine how the value vectors of `i0` and `i1` are mixed.

There is no causal mask because both instruments at a given timestep are simultaneously available.

### Restore original tensor shape

```python
x = x.reshape(B, T, A, D)
```

This converts:

```text
[B*T, A, D]
```

back to:

```text
[B, T, A, D]
```

### Residual connection

```python
return residual + x
```

The attention branch is treated as an additive update:

```text
X_new
=
X_old
+
Δ_attention
```

The returned tensor therefore still has shape:

```text
[B, T, A, D]
```

---

# `TemporalGRU`

```python
class TemporalGRU(nn.Module):
```

`TemporalGRU` performs recurrent processing through time independently for each instrument.

Conceptually:

```text
i0:
t0 → t1 → t2 → ... → t511

i1:
t0 → t1 → t2 → ... → t511
```

Unlike `AssetMixer`, which operates between instruments at one timestep, this module operates along the temporal dimension.

## Initialization

```python
def __init__(
    self,
    d_model=64,
    rnn_multiplier=4,
):
```

The input width is:

```text
64
```

The GRU hidden width is calculated as:

```python
hidden_dim = d_model * rnn_multiplier
```

With the defaults:

```text
64 × 4 = 256
```

Thus:

```text
GRU input width  = 64
GRU hidden width = 256
```

## Layer normalization

```python
self.norm = nn.LayerNorm(d_model)
```

Each 64-dimensional representation is normalized before entering the GRU.

## GRU definition

```python
self.gru = nn.GRU(
    input_size=d_model,
    hidden_size=hidden_dim,
    batch_first=True,
)
```

Each temporal step provides a 64-dimensional input.

The GRU maintains a 256-dimensional recurrent hidden state.

Conceptually:

```text
x_t       [64]
h_(t-1)  [256]
   │
   ▼
  GRU
   │
   ▼
h_t      [256]
```

## Projection back to `d_model`

```python
self.projection = nn.Linear(
    hidden_dim,
    d_model,
)
```

The GRU produces 256-dimensional outputs.

The rest of the model operates with:

```text
D = 64
```

so the GRU output is projected:

```text
256 → 64
```

## `TemporalGRU.forward()`

The input has shape:

```text
[B, T, A, D]
```

For example:

```text
[1, 512, 2, 64]
```

### Preserve residual

```python
residual = x
```

As with attention, the temporal module learns an additive update rather than replacing the representation entirely.

### Normalize

```python
x = self.norm(x)
```

Shape remains:

```text
[B, T, A, D]
```

### Move the asset dimension before time

```python
x = x.permute(0, 2, 1, 3)
```

This changes:

```text
[B, T, A, D]
```

into:

```text
[B, A, T, D]
```

For example:

```text
[1, 512, 2, 64]
    ↓
[1, 2, 512, 64]
```

No data values are changed.

Only the interpretation/order of the axes changes.

### Flatten batch and asset dimensions

```python
x = x.reshape(B * A, T, D)
```

For:

```text
B = 1
A = 2
```

this produces:

```text
[2, 512, 64]
```

The GRU therefore sees:

```text
2 independent sequences
512 timesteps per sequence
64 features per timestep
```

Those two sequences are:

```text
sequence 0:
    i0 through time

sequence 1:
    i1 through time
```

This reshape is what mechanically makes the GRU operate through time.

### Run the GRU

```python
x, hidden = self.gru(
    x,
    hidden,
)
```

The GRU processes each temporal sequence.

The output `x` contains the GRU output at every timestep.

With the default hidden size:

```text
input:
[2, 512, 64]

GRU output:
[2, 512, 256]
```

`hidden` contains only the final GRU state for each sequence.

Its shape is approximately:

```text
[1, B*A, 256]
```

The leading `1` is the number of GRU layers.

The returned hidden state is used by the training/evaluation code to continue the same GRU through the next temporal chunk.

It is not passed from one depth block to the next.

### Project back to 64 dimensions

```python
x = self.projection(x)
```

This performs:

```text
[2,512,256]
    ↓
[2,512,64]
```

### Restore original axes

```python
x = x.reshape(B, A, T, D)
x = x.permute(0, 2, 1, 3)
```

This returns the tensor to:

```text
[B, T, A, D]
```

For example:

```text
[1, 512, 2, 64]
```

### Residual connection

```python
return residual + x, hidden
```

The temporal update is added to the representation that entered the GRU module:

```text
X_new
=
X_old
+
Δ_temporal
```

Two objects are returned:

```text
updated representation
updated GRU hidden state
```

---

# `FeedForward`

```python
class FeedForward(nn.Module):
```

The feed-forward network differs from the two previous modules because it does not exchange information between assets or between timesteps.

Instead, it independently transforms every 64-dimensional representation.

## Expansion

```python
hidden_dim = d_model * expansion
```

With:

```text
d_model = 64
expansion = 4
```

the FFN uses:

```text
64 → 256 → 64
```

## Network

```python
self.net = nn.Sequential(
    nn.Linear(d_model, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, d_model),
)
```

For each individual latent vector:

```text
[64]
 ↓
Linear
 ↓
[256]
 ↓
GELU
 ↓
[256]
 ↓
Linear
 ↓
[64]
```

`GELU` supplies the nonlinearity.

Without a nonlinear activation, consecutive linear layers could collapse mathematically into one linear transformation.

## Forward pass

```python
return x + self.net(self.norm(x))
```

Again the module uses:

```text
LayerNorm
+
learned update
+
residual connection
```

Thus:

```text
X_new
=
X_old
+
Δ_FFN
```

The outer tensor shape remains:

```text
[B, T, A, D]
```

---

# `AxialGRUBlock`

```python
class AxialGRUBlock(nn.Module):
```

One complete axial block combines the previous three modules:

```text
AssetMixer
    ↓
TemporalGRU
    ↓
FeedForward
```

The constructor creates one instance of each:

```python
self.asset_mixer = AssetMixer(...)

self.temporal_mixer = TemporalGRU(...)

self.ffn = FeedForward(...)
```

## Forward pass

```python
x = self.asset_mixer(x)

x, hidden = self.temporal_mixer(
    x,
    hidden,
)

x = self.ffn(x)

return x, hidden
```

Thus one block performs:

```text
1. Cross-instrument mixing
2. Temporal mixing
3. Nonlinear per-representation refinement
```

All three stages preserve:

```text
[B, T, A, D]
```

The GRU additionally produces a final recurrent hidden state.

---

# Why this is called axial

The input has two meaningful structural axes:

```text
time × instrument
```

Conceptually:

```text
                 instrument axis
              i0              i1

time 0       [vector]       [vector]
time 1       [vector]       [vector]
time 2       [vector]       [vector]
 ...            ...            ...
```

The model handles the two axes separately.

The asset axis uses:

```text
self-attention
```

The temporal axis uses:

```text
GRU recurrence
```

This separate processing of different axes is the core axial idea.

Patrick's modification differs from a conventional axial Transformer because the temporal axis is handled using a GRU rather than another self-attention module.

---

# `count_parameters()`

```python
def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
```

This is a helper function.

It counts every trainable scalar parameter in a model.

```python
model.parameters()
```

iterates over the model parameters.

```python
p.numel()
```

counts the number of scalar values in one parameter tensor.

```python
p.requires_grad
```

ensures only trainable parameters are counted.

This helper is used by the training code to report model size.

---

# `WunderFeatureEncoder`

```python
class WunderFeatureEncoder(nn.Module):
```

The raw Wunder input for one instrument has:

```text
60 features
```

The model uses an internal width:

```text
d_model = 64
```

The encoder therefore performs:

```text
60 → 64
```

## Projection

```python
self.projection = nn.Linear(
    input_dim,
    d_model,
)
```

This defines a learned linear transformation.

For one feature vector:

```text
x [60]
```

the encoder computes:

```text
y = Wx + b
```

producing:

```text
y [64]
```

## Forward pass

```python
def forward(self, x):
    return self.projection(x)
```

`nn.Linear` operates automatically on the last tensor dimension.

Thus:

```text
[B,T,A,60]
    ↓
[B,T,A,64]
```

No asset mixing or temporal processing happens here.

This simply places the raw Wunder features into the common latent representation used by the rest of the model.

---

# Model 1: `WunderAxialGRUModel`

```python
class WunderAxialGRUModel(nn.Module):
```

This is the baseline model.

It uses several independently parameterized axial blocks.

With the default:

```text
num_blocks = 3
```

the architecture is:

```text
FeatureEncoder
      ↓
Block 1  parameters θ1
      ↓
Block 2  parameters θ2
      ↓
Block 3  parameters θ3
      ↓
Prediction head
```

Every block has its own:

```text
AssetMixer
TemporalGRU
FeedForward
```

and therefore its own trainable parameters.

## Create independent blocks

```python
self.blocks = nn.ModuleList([
    AxialGRUBlock(...)
    for _ in range(num_blocks)
])
```

With three blocks this creates three distinct Python/PyTorch module objects.

Thus:

```text
θ1 ≠ θ2 ≠ θ3
```

## Prediction head

```python
self.head = nn.Linear(
    d_model,
    num_targets,
)
```

With the defaults:

```text
64 → 2
```

The two outputs are:

```text
predicted t0
predicted t1
```

Both targets relate to instrument `i0`.

## Hidden states

Every block contains its own GRU.

Therefore each block also has its own temporal hidden state.

For three blocks:

```text
hidden_states = [
    h_block1,
    h_block2,
    h_block3,
]
```

If there are no previous states:

```python
hidden_states = [
    None
    for _ in self.blocks
]
```

Each GRU therefore starts from a fresh initial state.

## Run blocks

```python
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
```

The latent representation flows through depth:

```text
x0
 ↓
Block 1
 ↓
x1
 ↓
Block 2
 ↓
x2
 ↓
Block 3
 ↓
x3
```

The GRU hidden states do not flow from one block to another.

Instead:

```text
Block 1:
h1_old → h1_new

Block 2:
h2_old → h2_new

Block 3:
h3_old → h3_new
```

The updated hidden states are returned to the external training or evaluation loop.

That external code passes them back into the same depth positions when processing the next temporal chunk.

## Select instrument `i0`

After all blocks:

```python
target_asset = x[:, :, 0, :]
```

At this point:

```text
x: [B,T,2,64]
```

The indexing selects:

```text
all batches
all timesteps
asset index 0 = i0
all 64 latent dimensions
```

Thus:

```text
[B,T,2,64]
    ↓
[B,T,64]
```

Instrument `i1` is not directly predicted.

Its information has already influenced the representation of `i0` through the AssetMixer operations.

## Produce predictions

```python
predictions = self.head(
    target_asset
)
```

This performs:

```text
[B,T,64]
    ↓
[B,T,2]
```

The final dimension contains:

```text
prediction[...,0] = predicted t0
prediction[...,1] = predicted t1
```

The size `2` here refers to the two targets, not the two instruments.

---

# Model 2: `WunderLoopedAxialGRUModel`

```python
class WunderLoopedAxialGRUModel(nn.Module):
```

This is the weight-tied / recurrent-depth variant.

The architecture is nearly identical to the baseline except that it creates:

```text
one AxialGRUBlock
```

rather than:

```text
multiple independent AxialGRUBlocks
```

## One shared block

```python
self.block = AxialGRUBlock(...)
```

Only one set of block parameters exists.

If:

```text
num_iterations = 3
```

the same block is executed three times:

```text
x0
 ↓
Block θ
 ↓
x1
 ↓
same Block θ
 ↓
x2
 ↓
same Block θ
 ↓
x3
```

Mathematically:

```text
H1 = Bθ(H0)
H2 = Bθ(H1)
H3 = Bθ(H2)
```

The same parameters `θ` are reused at every depth iteration.

## Iteration loop

```python
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
```

The representation `x` passes from one iteration to the next.

The block weights remain identical.

## Hidden states

Although the block weights are shared, each depth iteration maintains its own temporal hidden state.

For three iterations:

```text
hidden_states[0]
hidden_states[1]
hidden_states[2]
```

These represent the temporal state corresponding to each recurrent-depth position.

They are returned after the current chunk and reused by the same depth positions on the next temporal chunk.

## Main difference from baseline

Baseline:

```text
Block θ1
 ↓
Block θ2
 ↓
Block θ3
```

Looped:

```text
Block θ
 ↓
Block θ
 ↓
Block θ
```

Thus the looped model performs roughly the same number of block executions but stores far fewer trainable parameters.

With the current defaults:

```text
baseline:
~945,602 parameters

looped:
~317,890 parameters
```

The looped model therefore reduces parameter count substantially without reducing the number of refinement passes.

---

# Model 3: `WunderReinjectedLoopedAxialGRUModel`

```python
class WunderReinjectedLoopedAxialGRUModel(nn.Module):
```

This model begins with the same weight-tied architecture as `WunderLoopedAxialGRUModel`.

Its additional idea is:

> before later refinement iterations, mix the current representation with the original encoded input.

## Preserve original encoded input

Instead of:

```python
x = self.encoder(x)
```

the model uses:

```python
h0 = self.encoder(x)

x = h0
```

`h0` is therefore the original encoded observation:

```text
[B,T,A,64]
```

It is not a GRU hidden state.

It remains unchanged during the recurrent-depth refinement process.

---

# Reinjection gate

```python
self.reinjection_logit = nn.Parameter(
    torch.tensor(-2.0)
)
```

This introduces one additional trainable scalar parameter.

The forward pass converts it into a value between 0 and 1:

```python
gate = torch.sigmoid(
    self.reinjection_logit
)
```

Initially:

```text
reinjection_logit = -2
```

therefore:

```text
gate ≈ 0.1192
```

because:

```text
sigmoid(-2)
≈ 0.1192
```

The value `-2` is an initialization choice rather than something derived from the dataset.

## What the gate controls

For iterations after the first:

```python
x = (
    (1.0 - gate) * x
    + gate * h0
)
```

If:

```text
gate = 0.12
```

the representation becomes approximately:

```text
0.88 × current representation
+
0.12 × original encoded representation
```

This allows later refinement passes to retain direct access to the original evidence.

---

# Iteration behavior

Suppose:

```text
num_iterations = 3
```

and:

```text
H0 = Encoder(X)
```

## Iteration 1

The first iteration already receives `H0`, so no reinjection is needed:

```text
H1 = Block(H0)
```

## Iteration 2

Mix:

```text
H1_mixed
=
(1-g)H1 + gH0
```

then:

```text
H2
=
Block(H1_mixed)
```

## Iteration 3

Mix again:

```text
H2_mixed
=
(1-g)H2 + gH0
```

then:

```text
H3
=
Block(H2_mixed)
```

Thus:

```text
            H0 ─────────────┐
                            ▼
H0 → Block → H1 → mix → Block → H2 → mix → Block → H3
                                      ▲
            H0 ───────────────────────┘
```

The same AxialGRUBlock parameters are reused at every iteration.

---

# How the gate is trained

`reinjection_logit` is an:

```python
nn.Parameter
```

so PyTorch automatically includes it in:

```python
model.parameters()
```

The optimizer therefore updates it along with all other trainable model parameters.

During one forward pass:

```text
gate remains fixed
```

across all recurrent-depth iterations.

After the loss is calculated:

```text
loss.backward()
optimizer.step()
```

may modify `reinjection_logit`.

On the next forward pass:

```python
gate = torch.sigmoid(
    self.reinjection_logit
)
```

uses the newly updated value.

Thus the gate changes only when the optimizer performs a training update.

It does not change between depth iterations within the same forward pass.

---

# Shared versus possible future gate designs

The current implementation uses:

```text
one scalar gate
```

shared across:

```text
all timesteps
both instruments
all latent dimensions
all recurrent-depth iterations
```

This makes the current experiment extremely simple.

Possible future variants could include:

```text
one gate per depth iteration
different gates per latent dimension
context-dependent / dynamic gates
```

but these are not implemented in the current model.

---

# Parameter comparison

With the current default architecture:

```text
WunderAxialGRUModel
    ~945,602 parameters

WunderLoopedAxialGRUModel
    ~317,890 parameters

WunderReinjectedLoopedAxialGRUModel
    ~317,891 parameters
```

The reinjected model therefore adds only one parameter relative to the plain looped model:

```text
reinjection_logit
```

---

# Relationship between depth and time

The model has two distinct forms of progression that should not be confused.

## Depth

Within one chunk:

```text
x0
 ↓
Block / iteration 1
 ↓
x1
 ↓
Block / iteration 2
 ↓
x2
 ↓
Block / iteration 3
 ↓
x3
```

This represents model depth / refinement.

## Time

Inside each GRU:

```text
t0 → t1 → t2 → ... → t511
```

The GRU hidden state stores temporal information.

When the sequence is processed in chunks, the external training/evaluation code carries these hidden states from one chunk to the next.

Thus:

```text
representation x
    → flows through model depth

GRU hidden state
    → flows through time
```

These are separate concepts.

---

# Chunking

`axial_gru.py` itself does not divide 20,000-step sequences into chunks.

Chunking is handled externally by:

```text
train_wunder.py
evaluate_wunder.py
```

A complete sequence might be processed as:

```text
steps 0–511
   ↓ carry GRU hidden states
steps 512–1023
   ↓ carry GRU hidden states
steps 1024–1535
   ↓
...
```

The model receives one chunk per forward call.

The updated GRU states returned by the model are passed into the next chunk by the external training or evaluation code.

---

# Overall model flow

```text
Wunder input
[B,T,2,60]
       │
       ▼
WunderFeatureEncoder
60 → 64
       │
       ▼
[B,T,2,64]
       │
       ▼
┌─────────────────────────────┐
│        AxialGRUBlock        │
│                             │
│  AssetMixer                 │
│      i0 ↔ i1                │
│         ↓                   │
│  TemporalGRU                │
│      t0 → t1 → ...          │
│         ↓                   │
│  FeedForward                │
└─────────────────────────────┘
       │
       │ repeated according to model type
       ▼
final representation
[B,T,2,64]
       │
       │ select asset i0
       ▼
[B,T,64]
       │
       ▼
Linear(64 → 2)
       │
       ▼
[B,T,2]
       │
       ├── predicted t0
       └── predicted t1
```

The final `2` represents the two prediction targets.

It does not represent the two input instruments.

---

# The three experimental models

## Baseline

```text
Encoder
 ↓
Block θ1
 ↓
Block θ2
 ↓
Block θ3
 ↓
Prediction
```

Each depth has separate parameters.

## Looped

```text
Encoder
 ↓
Block θ
 ↓
same Block θ
 ↓
same Block θ
 ↓
Prediction
```

The block parameters are shared across depth.

## Reinjected looped

```text
Encoder → H0
           │
           ▼
        Block θ
           │
           ▼
          H1
           │
       mix with H0
           │
           ▼
        Block θ
           │
           ▼
          H2
           │
       mix with H0
           │
           ▼
        Block θ
           │
           ▼
       Prediction
```

This preserves the parameter savings of recurrent depth while allowing later iterations to revisit the original encoded observation.

---

# Research idea represented by this file

The baseline asks:

> What happens when each depth has its own independently learned transformation?

The looped model asks:

> Can the same transformation be applied repeatedly to refine the representation while greatly reducing parameter count?

The reinjected model asks:

> If aggressive weight sharing loses information or performance, can repeatedly reintroducing the original encoded observation recover that performance with essentially no additional parameters?

This gives the core experimental comparison:

```text
independent depth
vs.
shared recurrent depth
vs.
shared recurrent depth + source reinjection
```

---

# In one sentence

`axial_gru.py` defines a dual-axis financial time-series model that mixes information across instruments using self-attention, models temporal history using a GRU, refines each representation with an FFN, and provides baseline, weight-tied recurrent-depth, and source-reinjected recurrent-depth variants for controlled comparison.
