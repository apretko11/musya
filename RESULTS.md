# Wunder Axial-GRU Preliminary Results

## Status

These are preliminary results from the first controlled comparison of three architectures on the Wunder Connectome dataset:

1. **Baseline** — three independently parameterized Axial-GRU blocks.
2. **Looped** — one Axial-GRU block reused for three recurrent-depth iterations.
3. **Reinjected looped** — same shared recurrent-depth block, with learned reinjection of the original encoded input before later iterations.

The main question is whether recurrent-depth parameter sharing can preserve predictive performance while substantially reducing the number of learned parameters, and whether source reinjection can recover or improve performance under weight sharing.

These results should be treated as an initial experiment rather than final paper results.

---

# Experimental setup

All three models use the same basic block:

```text
Asset self-attention
        ↓
Temporal GRU
        ↓
Feed-forward network
```

Default dimensions:

```text
Input per instrument:     60 features
Model dimension:          64
Attention heads:          4
GRU multiplier:           4
Depth / iterations:       3
Chunk size:               512
```

Training configuration:

```text
Training sequences:       100
Internal validation:      20 sequences
Epochs:                   3
Model seeds:              0, 1, 2
Split seed:               1234
Optimizer:                Adam
Learning rate:            1e-4
```

The same train/validation sequence split was used for all architectures and model seeds.

Hardware:

```text
GPU:                      NVIDIA RTX 5000 Ada Generation
GPU memory:               ~32 GB
PyTorch:                  2.12.0+cu126
CUDA runtime:             12.6
```

The best checkpoint for each run was selected using internal validation MSE.

Each best checkpoint was then evaluated over the complete Wunder validation set:

```text
Validation sequences:     1,873
Scored rows:              3,528,273
```

The primary reported evaluation metric is Wunder's Global Weighted Pearson correlation.

---

# Model sizes

| Model | Trainable parameters | Parameter memory (FP32) | Reduction vs. baseline |
|---|---:|---:|---:|
| Baseline | 945,602 | 3.607 MiB | — |
| Looped | 317,890 | 1.213 MiB | 66.4% |
| Reinjected | 317,891 | 1.213 MiB | 66.4% |

The looped architecture therefore uses approximately one-third as many trainable parameters as the baseline while retaining three block executions.

The reinjected architecture adds only one trainable scalar parameter relative to the plain looped model.

---

# Internal validation MSE

## Baseline

```text
Seed 0: 1.603287
Seed 1: 1.616629
Seed 2: 1.611235
```

## Looped

```text
Seed 0: 1.614550
Seed 1: 1.611696
Seed 2: 1.620297
```

## Reinjected

```text
Seed 0: 1.613851
Seed 1: 1.612738
Seed 2: 1.621159
```

The internal validation MSE values are broadly similar across architectures.

These values were used for checkpoint selection, but the primary evaluation metric for the task is Weighted Pearson rather than MSE.

---

# Full Wunder validation results

## Baseline

| Seed | t0 | t1 | Global Weighted Pearson |
|---|---:|---:|---:|
| 0 | 0.56192499 | 0.56308761 | 0.56250630 |
| 1 | 0.47787860 | 0.52279898 | 0.50033879 |
| 2 | 0.55603815 | 0.56537836 | 0.56070826 |

```text
Mean Weighted Pearson: 0.54118445
Standard deviation:    0.03538480
```

---

## Looped

| Seed | t0 | t1 | Global Weighted Pearson |
|---|---:|---:|---:|
| 0 | 0.51969635 | 0.54533678 | 0.53251656 |
| 1 | 0.53642966 | 0.56547854 | 0.55095410 |
| 2 | 0.54655214 | 0.56381882 | 0.55518548 |

```text
Mean Weighted Pearson: 0.54621872
Standard deviation:    0.01205354
```

---

## Reinjected looped

| Seed | t0 | t1 | Global Weighted Pearson |
|---|---:|---:|---:|
| 0 | 0.52201210 | 0.54762816 | 0.53482013 |
| 1 | 0.53677109 | 0.56476555 | 0.55076832 |
| 2 | 0.54900103 | 0.56804401 | 0.55852252 |

```text
Mean Weighted Pearson: 0.54803699
Standard deviation:    0.01208495
```

---

# Summary comparison

| Model | Parameters | Weighted Pearson mean | Std. dev. |
|---|---:|---:|---:|
| Baseline | 945,602 | 0.54118445 | 0.03538480 |
| Looped | 317,890 | 0.54621872 | 0.01205354 |
| Reinjected | 317,891 | 0.54803699 | 0.01208495 |

Differences in mean score:

```text
Looped - Baseline:
+0.00503427

Reinjected - Baseline:
+0.00685254

Reinjected - Looped:
+0.00181827
```

At this stage, these differences should be treated descriptively rather than as statistically established improvements because only three model seeds have been evaluated.

---

# Reinjection gate

The reinjected model initializes its scalar gate as:

```text
sigmoid(-2) ≈ 0.1192
```

The learned gate values from the three runs were:

```text
Seed 0: 0.137848
Seed 1: 0.142629
Seed 2: 0.143389
```

All three runs moved the gate upward from its initialization.

This indicates that optimization consistently learned to increase the contribution of the original encoded input beyond the initial ~11.9%.

The gate remains relatively small, however, so the model still primarily uses the current recurrent-depth representation rather than replacing it with the original input.

---

# Training runtime

Total runtime for the three-epoch training runs:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Baseline | 335.2 s | 412.2 s | 346.9 s |
| Looped | 326.5 s | 350.9 s | 334.7 s |
| Reinjected | 339.2 s | 330.3 s | 343.9 s |

There is no evidence from these initial measurements of a major runtime reduction from parameter sharing.

This is expected because all three architectures still execute approximately the same number of Axial-GRU block passes.

Weight sharing reduces the number of stored parameters, not the number of block computations.

---

# GPU memory benchmark

Memory was measured using a real Wunder sequence with:

```text
Chunk size:         512
Batch size:         1
Datatype:           FP32
Optimizer:          Adam
```

For training memory, an initial optimizer step was performed before measurement so that Adam's optimizer-state buffers were allocated.

## Measured memory

| Model | Parameter memory | Inference peak allocated | Training peak allocated |
|---|---:|---:|---:|
| Baseline | 3.607 MiB | 25.61 MiB | 89.86 MiB |
| Looped | 1.213 MiB | 31.34 MiB | 82.66 MiB |
| Reinjected | 1.213 MiB | 31.59 MiB | 83.16 MiB |

Peak reserved memory:

| Model | Inference reserved | Training reserved |
|---|---:|---:|
| Baseline | 28.00 MiB | 138.00 MiB |
| Looped | 46.00 MiB | 136.00 MiB |
| Reinjected | 46.00 MiB | 136.00 MiB |

The parameter-memory reduction is large and deterministic:

```text
Baseline:      3.607 MiB
Looped:        1.213 MiB
Reinjected:    1.213 MiB
```

However, total GPU memory does not decrease proportionally because model weights are only a small portion of the total runtime memory footprint.

Training activations, recurrent intermediates, gradients, optimizer state, and CUDA workspaces dominate memory at this model scale.

The looped models showed a modest reduction in measured peak training allocation:

```text
Baseline:      89.86 MiB
Looped:        82.66 MiB
Reinjected:    83.16 MiB
```

Unexpectedly, the looped models showed somewhat higher peak inference allocation:

```text
Baseline:      25.61 MiB
Looped:        31.34 MiB
Reinjected:    31.59 MiB
```

Because this inference-memory difference is small in absolute terms and currently comes from one benchmark run, it should be replicated before drawing conclusions.

---

# Preliminary conclusions

## 1. Recurrent depth dramatically reduces trainable parameter count

Replacing three independent Axial-GRU blocks with one repeatedly applied shared block reduced trainable parameters from:

```text
945,602
```

to:

```text
317,890
```

which is approximately a:

```text
66.4% reduction
```

This is the clearest result from the current experiment.

---

## 2. The parameter reduction did not reduce mean predictive performance

Despite using approximately one-third as many parameters, the looped model achieved:

```text
Baseline mean WP: 0.54118
Looped mean WP:   0.54622
```

The reinjected model achieved:

```text
Reinjected mean WP: 0.54804
```

Thus, in this initial experiment, aggressive recurrent-depth parameter sharing preserved mean predictive performance.

The current data does not justify claiming a statistically significant performance improvement.

---

## 3. The shared-depth models were more consistent across these three seeds

Observed standard deviations were:

```text
Baseline:      0.03538
Looped:        0.01205
Reinjected:    0.01208
```

The baseline's larger variance is driven in particular by seed 1, which scored substantially below seeds 0 and 2.

This is potentially interesting, but three seeds are insufficient to conclude that weight sharing genuinely improves training stability.

More seeds are required.

---

## 4. Reinjection produced a small additional improvement

Mean Weighted Pearson:

```text
Looped:        0.54622
Reinjected:    0.54804
```

The difference is:

```text
+0.00182
```

This is in the desired direction but is currently too small relative to seed variation to support a strong claim.

The learned gates nevertheless provide evidence that the model actively uses the reinjection mechanism.

All three gates increased from the initial value of approximately 0.1192 to approximately 0.138-0.143.

---

## 5. Fewer parameters does not imply proportionally lower runtime GPU memory

The looped models reduce model-weight storage by approximately two-thirds, but activation and optimizer memory dominate the current small architecture.

As a result:

```text
Parameter memory:
~66% lower

Peak training GPU allocation:
only modestly lower
```

Inference GPU memory was actually somewhat higher for the shared-depth models in the initial benchmark.

This distinction between parameter efficiency and runtime memory efficiency should be maintained explicitly.

---

# Current interpretation

The strongest statement supported by the experiment so far is:

> A three-pass recurrent-depth Axial-GRU using one shared block achieved comparable or slightly higher mean Wunder validation performance than a three-block independently parameterized baseline while using approximately 66% fewer trainable parameters.

A secondary preliminary observation is:

> Reinjection of the original encoded input produced a small additional improvement over plain recurrent depth, and the learned gate consistently increased above its initialization across all three seeds.

Neither the performance improvement nor the apparent reduction in seed sensitivity should yet be treated as established conclusions.

---

# Next experimental questions

The current results motivate several follow-up experiments:

1. Increase the number of random seeds to determine whether the mean and variance differences are robust.

2. Test recurrent-depth iteration counts such as:

```text
K = 1
K = 2
K = 3
K = 4
```

to determine the relationship between iterative refinement and predictive performance.

3. Compare reinjection strategies:
   - no reinjection
   - fixed reinjection gate
   - one learned global gate
   - one learned gate per recurrent-depth iteration
   - potentially dynamic/context-dependent gating

4. Repeat the GPU memory benchmark several times to verify the unexpected inference-memory result.

5. Test parameter-matched controls so that improvements can be attributed more clearly to recurrent depth rather than merely differences in effective capacity.

6. Train on a larger portion of the available Wunder training set.

7. Evaluate whether the current conclusions hold at larger model widths.

8. Test a second financial time-series dataset if available to determine whether the effect generalizes beyond Wunder.

---

# Short version for discussion

The first experiment is promising.

Using one Axial-GRU block repeatedly instead of three separate blocks reduces the parameter count by approximately 66%.

Despite this reduction, the three-seed mean Weighted Pearson score was slightly higher:

```text
Baseline:      0.5412 ± 0.0354
Looped:        0.5462 ± 0.0121
Reinjected:    0.5480 ± 0.0121
```

The reinjected version performed slightly better than the plain looped model, although the difference is currently small.

The shared-depth models also showed much lower variance across the three seeds, but more seeds are needed before interpreting this as improved stability.

The reduction in parameter count does not translate directly into an equivalent GPU-memory reduction because activations and training state dominate memory for this relatively small model.

Overall, the initial result supports continuing the recurrent-depth direction and justifies a broader ablation study.
