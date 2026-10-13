# Wunder Axial-GRU Experimental Results

## Status

This document summarizes the current experimental results for a study of recurrent/shared depth in an axial financial time-series model.

The model operates over two structural axes:

1. an instrument axis, modeled using self-attention; and
2. a temporal axis, modeled using a GRU.

One Axial-GRU block is:

```text
Asset self-attention
        ↓
Temporal GRU
        ↓
Feed-forward network
```

Three model families are compared:

1. **Baseline** — multiple independently parameterized Axial-GRU blocks.
2. **Looped** — one Axial-GRU block repeatedly reused across recurrent depth.
3. **Reinjected looped** — the same shared block, with learned reinjection of the original encoded input before recurrent iterations after the first.

The original hypothesis was that recurrent-depth weight sharing might preserve predictive performance while substantially reducing the number of learned parameters, and that source reinjection might recover information lost under repeated weight-tied transformations.

The current results refine that hypothesis considerably.

The strongest evidence so far is not that recurrent depth outperforms a shallow parameter-matched model. Instead:

> Repeated application of a shared block progressively degrades predictive performance under the current setup, while reinjection of the original encoded representation consistently mitigates that degradation. The mitigation becomes larger at greater recurrent depth.

These results remain preliminary until the larger-data confirmation and final efficiency measurements are complete.

---

# Dataset and task

Experiments use the Wunder Connectome dataset.

Each sequence contains:

```text
20,000 timesteps
2 instruments: i0 and i1
60 input features per instrument after preprocessing
2 prediction targets: t0 and t1
```

The model input therefore has shape:

```text
[T, 2, 60]
```

and predictions have shape:

```text
[T, 2]
```

The two output dimensions correspond to `t0` and `t1`, not to the two instruments.

Both targets concern instrument `i0`; instrument `i1` provides contextual information.

The primary final-evaluation metric is Wunder's Global Weighted Pearson correlation.

Full official validation contains:

```text
1,873 sequences
3,528,273 scored rows
```

---

# Core architecture

Default model dimensions:

```text
Input features per instrument: 60
Model dimension:               64
Attention heads:               4
GRU multiplier:                4
Training chunk size:           512
```

A complete Axial-GRU block performs:

```text
1. cross-instrument self-attention
2. temporal GRU processing
3. per-representation feed-forward transformation
```

Residual connections are used inside each stage.

---

# Model families

## Independent baseline

For depth N:

```text
Encoder
   ↓
Block θ1
   ↓
Block θ2
   ↓
...
   ↓
Block θN
   ↓
Prediction head
```

Each block has independent parameters.

---

## Looped recurrent depth

For recurrent depth K:

```text
Encoder
   ↓
Block θ
   ↓
same Block θ
   ↓
...
   ↓
same Block θ
   ↓
Prediction head
```

The same block parameters are reused for every recurrent-depth iteration.

---

## Reinjected recurrent depth

The reinjected model preserves the original encoded representation:

```text
H0 = Encoder(X)
```

The first recurrent-depth iteration receives `H0` directly.

For later iterations:

```text
H_mixed = (1 - g) H_current + g H0
```

where:

```text
g = sigmoid(reinjection_logit)
```

and `reinjection_logit` is a learned scalar parameter.

The mixed representation is then passed through the same shared Axial-GRU block.

---

# Experimental protocol

Unless otherwise specified:

```text
Training sequences:             100
Internal validation sequences:  20
Epochs:                         3
Split seed:                     1234
Optimizer:                      Adam
Learning rate:                  1e-4
Training chunk size:            512
```

The same train/internal-validation split is used across compared configurations.

The best checkpoint is selected using internal validation MSE.

That checkpoint is then evaluated over all 1,873 official validation sequences using Global Weighted Pearson.

Official validation inference was initially performed with chunk size 512.

A later chunk-size benchmark showed that larger evaluation chunks produce effectively identical predictions and scores while reducing evaluation time.

The evaluation chunk size used for subsequent experiments is:

```text
8192
```

---

# Evaluation chunk-size benchmark

One checkpoint was evaluated over the same 100 official validation sequences using four inference chunk sizes.

| Chunk size | Weighted Pearson | Time |
|---:|---:|---:|
| 512 | 0.5470113284 | 52.32 s |
| 2,048 | 0.5470113426 | 40.94 s |
| 8,192 | 0.5470113594 | 39.95 s |
| 20,000 | 0.5470113587 | 41.42 s |

Differences in Weighted Pearson are on the order of approximately:

```text
10^-8
```

and are therefore negligible for the present experiments.

Chunk size 8192 was selected for later full validation runs.

---

# Hardware

Experiments have been executed on two NVIDIA GPUs.

## Campus GPU

```text
NVIDIA RTX 5000 Ada Generation
VRAM: approximately 32 GB
Driver: 570.195.03
PyTorch: 2.12.0+cu126
CUDA runtime: 12.6
```

## Hyperstack GPU

```text
NVIDIA RTX A4000
VRAM: approximately 15 GB
Driver: 570.195.03
PyTorch: 2.12.0+cu126
CUDA runtime: 12.6
```

A matched smoke experiment produced effectively identical losses on the two GPUs, with only very small floating-point differences.

The A4000 was approximately 1.6x slower for the small smoke-training run.

Final efficiency comparisons should therefore be performed on one consistent GPU rather than comparing raw runtime measurements across the two systems.

---

# Parameter counts

One Axial-GRU block plus the common encoder/head uses approximately:

```text
317,890 trainable parameters
```

The independently parameterized baseline grows with depth.

| Configuration | Trainable parameters |
|---|---:|
| Baseline N=1 | 317,890 |
| Baseline N=2 | 631,746 |
| Baseline N=3 | 945,602 |
| Baseline N=4 | 1,259,458 |
| Looped K=1 | 317,890 |
| Looped K=2 | 317,890 |
| Looped K=3 | 317,890 |
| Looped K=4 | 317,890 |
| Reinjected K=1 | 317,891 |
| Reinjected K=2 | 317,891 |
| Reinjected K=3 | 317,891 |
| Reinjected K=4 | 317,891 |

The reinjected architecture adds only one scalar trainable parameter relative to the plain looped model.

At depth three:

```text
Baseline N=3:   945,602 parameters
Looped K=3:     317,890 parameters
Reinjected K=3: 317,891 parameters
```

Thus recurrent-depth sharing reduces trainable parameters by approximately:

```text
66.4%
```

relative to the three-block independent baseline.

---

# Ten-seed main comparison

The original experiment used only three seeds and initially suggested that recurrent weight sharing might preserve or slightly improve performance.

That result did not survive expansion to ten seeds.

All configurations below use depth/iterations = 3.

## Baseline N=3

| Seed | t0 | t1 | Weighted Pearson |
|---:|---:|---:|---:|
| 0 | 0.56192499 | 0.56308761 | 0.56250630 |
| 1 | 0.47787860 | 0.52279898 | 0.50033879 |
| 2 | 0.55603815 | 0.56537836 | 0.56070826 |
| 3 | 0.56575169 | 0.57213811 | 0.56894490 |
| 4 | 0.56179482 | 0.56759532 | 0.56469507 |
| 5 | 0.57502521 | 0.54647978 | 0.56075250 |
| 6 | 0.55237630 | 0.56110694 | 0.55674162 |
| 7 | 0.54511729 | 0.57007052 | 0.55759390 |
| 8 | 0.56270053 | 0.55795561 | 0.56032807 |
| 9 | 0.55105445 | 0.56744887 | 0.55925166 |

```text
Mean Weighted Pearson: 0.55518611
Standard deviation:    0.01959082
Minimum:               0.50033879
Maximum:               0.56894490
```

---

## Looped K=3

| Seed | t0 | t1 | Weighted Pearson |
|---:|---:|---:|---:|
| 0 | 0.51969635 | 0.54533678 | 0.53251656 |
| 1 | 0.53642966 | 0.56547854 | 0.55095410 |
| 2 | 0.54655214 | 0.56381882 | 0.55518548 |
| 3 | 0.45826621 | 0.54727605 | 0.50277113 |
| 4 | 0.54926504 | 0.55499271 | 0.55212887 |
| 5 | 0.55549897 | 0.55657101 | 0.55603499 |
| 6 | 0.50421041 | 0.52741479 | 0.51581260 |
| 7 | 0.39879038 | 0.53265778 | 0.46572408 |
| 8 | 0.35364899 | 0.51560229 | 0.43462564 |
| 9 | 0.53583805 | 0.56957859 | 0.55270832 |

```text
Mean Weighted Pearson: 0.52184618
Standard deviation:    0.04251588
Minimum:               0.43462564
Maximum:               0.55603499
```

---

## Reinjected K=3

| Seed | t0 | t1 | Weighted Pearson |
|---:|---:|---:|---:|
| 0 | 0.52201210 | 0.54762816 | 0.53482013 |
| 1 | 0.53677109 | 0.56476555 | 0.55076832 |
| 2 | 0.54900103 | 0.56804401 | 0.55852252 |
| 3 | 0.46446699 | 0.55102704 | 0.50774701 |
| 4 | 0.55487880 | 0.55820783 | 0.55654332 |
| 5 | 0.56017741 | 0.55987129 | 0.56002435 |
| 6 | 0.51044762 | 0.53078972 | 0.52061867 |
| 7 | 0.41720841 | 0.53406889 | 0.47563865 |
| 8 | 0.55158830 | 0.56171331 | 0.55665081 |
| 9 | 0.53749526 | 0.56905539 | 0.55327533 |

```text
Mean Weighted Pearson: 0.53746091
Standard deviation:    0.02810262
Minimum:               0.47563865
Maximum:               0.56002435
```

---

# Ten-seed summary

| Model | Parameters | Mean Weighted Pearson | Std. dev. |
|---|---:|---:|---:|
| Baseline N=3 | 945,602 | 0.55518611 | 0.01959082 |
| Looped K=3 | 317,890 | 0.52184618 | 0.04251588 |
| Reinjected K=3 | 317,891 | 0.53746091 | 0.02810262 |

The ten-seed study reverses the interpretation suggested by the original three-seed experiment.

At K=3:

```text
Baseline > Reinjected > Looped
```

on mean Weighted Pearson.

Plain recurrent depth also exhibits substantially larger seed-to-seed variation than the independent baseline.

Reinjection recovers part of the performance lost by naive recurrent depth, but does not fully close the gap to the independently parameterized baseline.

---

# Five-seed depth study

A targeted depth study was run using seeds 0-4.

The goal was to determine whether performance degradation depends systematically on recurrent depth.

---

## Baseline: independent depth

| Depth | Mean Weighted Pearson | Std. dev. |
|---:|---:|---:|
| N=1 | 0.56652162 | 0.00453565 |
| N=2 | 0.56762320 | 0.00205026 |
| N=3 | 0.55143866 | 0.02873071 |
| N=4 | 0.55112109 | 0.00940587 |

Per-seed Weighted Pearson:

```text
N=1:
0.56694662
0.57247483
0.56345885
0.56885076
0.56087701

N=2:
0.56774728
0.56767713
0.56924202
0.56923089
0.56421870

N=3:
0.56250630
0.50033879
0.56070826
0.56894490
0.56469507

N=4:
0.55453745
0.54405364
0.56089844
0.53862666
0.55748926
```

The strongest five-seed mean is currently:

```text
Baseline N=2:
0.56762320 ± 0.00205026
```

The two-block independent baseline is also highly stable across these five seeds.

Adding independent depth beyond two blocks does not improve performance under the current training regime.

---

# Looped recurrent depth

| Iterations | Mean Weighted Pearson | Std. dev. |
|---:|---:|---:|
| K=1 | 0.56652162 | 0.00453565 |
| K=2 | 0.56149092 | 0.00255069 |
| K=3 | 0.53871123 | 0.02197377 |
| K=4 | 0.52567327 | 0.02120812 |

Per-seed Weighted Pearson:

```text
K=1:
0.56694662
0.57247483
0.56345885
0.56885076
0.56087701

K=2:
0.56407580
0.55926242
0.55974831
0.56446204
0.55990601

K=3:
0.53251656
0.55095410
0.55518548
0.50277113
0.55212887

K=4:
0.52227850
0.52314043
0.54811502
0.49352890
0.54130352
```

K=1 is equivalent to a one-block independent baseline because the shared block is executed only once.

The pattern with increasing recurrent depth is monotonic in the five-seed mean:

```text
K=1: 0.56652
K=2: 0.56149
K=3: 0.53871
K=4: 0.52567
```

Thus, in the current architecture and optimization regime, repeatedly applying the same block does not provide useful additional depth.

Instead, predictive performance progressively deteriorates as recurrent depth increases.

---

# Reinjected recurrent depth

| Iterations | Mean Weighted Pearson | Std. dev. |
|---:|---:|---:|
| K=1 | 0.56652162 | 0.00453565 |
| K=2 | 0.56253948 | 0.00225412 |
| K=3 | 0.54168026 | 0.02112872 |
| K=4 | 0.53619779 | 0.02160521 |

Per-seed Weighted Pearson:

```text
K=1:
0.56694662
0.57247483
0.56345885
0.56885076
0.56087701

K=2:
0.56441069
0.56046738
0.56103247
0.56549100
0.56129587

K=3:
0.53482013
0.55076832
0.55852252
0.50774701
0.55654332

K=4:
0.52874025
0.54636670
0.55359965
0.50161546
0.55066692
```

Reinjection does not eliminate recurrent-depth degradation.

However, it consistently mitigates that degradation.

---

# Paired reinjection effect

Because looped and reinjected models are trained with matched seeds, the most direct comparison is:

```text
Weighted Pearson(reinjected)
-
Weighted Pearson(looped)
```

for the same seed and recurrent depth.

## K=2

```text
Seed 0: +0.00033489
Seed 1: +0.00120496
Seed 2: +0.00128416
Seed 3: +0.00102896
Seed 4: +0.00138985
```

```text
Mean improvement: +0.00104856
Std. dev.:        0.00042017
```

Reinjection improves K=2 performance in all five seeds.

---

## K=3

```text
Seed 0: +0.00230356
Seed 1: -0.00018578
Seed 2: +0.00333703
Seed 3: +0.00497588
Seed 4: +0.00441445
```

```text
Mean improvement: +0.00296903
Std. dev.:        0.00204008
```

Reinjection improves four of the five paired K=3 runs.

---

## K=4

```text
Seed 0: +0.00646175
Seed 1: +0.02322627
Seed 2: +0.00548463
Seed 3: +0.00808655
Seed 4: +0.00936340
```

```text
Mean improvement: +0.01052452
Std. dev.:        0.00725488
```

Reinjection improves K=4 performance in all five seeds.

---

# Reinjection effect versus recurrent depth

The mean paired reinjection advantage increases with recurrent depth:

```text
K=2: +0.00105
K=3: +0.00297
K=4: +0.01052
```

This is currently the most interesting positive architectural result.

The data are consistent with the interpretation that preserving access to the original encoded representation becomes increasingly useful as the same transformation is repeatedly reapplied.

However, the experiments do not yet identify the exact mechanism responsible for recurrent-depth degradation.

Terms such as:

```text
representation drift
over-refinement
information loss
optimization difficulty
```

should therefore be treated as hypotheses rather than established explanations.

---

# Parameter-efficiency interpretation

The depth study provides a direct parameter-matched comparison.

A one-block baseline and all plain looped models have essentially the same number of parameters:

```text
Baseline N=1: 317,890 parameters

Looped K=1:   317,890
Looped K=2:   317,890
Looped K=3:   317,890
Looped K=4:   317,890
```

Yet the five-seed results are:

```text
Baseline N=1: 0.56652
Looped K=2:   0.56149
Looped K=3:   0.53871
Looped K=4:   0.52567
```

Thus additional recurrent computation does not improve performance over simply applying the parameter-matched block once.

The current evidence therefore does not support the claim that recurrent depth itself improves the parameter/performance tradeoff.

---

# Comparison with the strongest baseline

The strongest current configuration is:

```text
Baseline N=2
631,746 parameters
Weighted Pearson:
0.56762320 ± 0.00205026
```

A single-block model uses approximately half as many parameters:

```text
Baseline N=1
317,890 parameters
Weighted Pearson:
0.56652162 ± 0.00453565
```

The performance difference between N=1 and N=2 is small in the current five-seed experiment.

This suggests that much of the useful predictive performance may already be obtained with a relatively shallow architecture.

That observation should be tested again in the larger-data experiment.

---

# Initial GPU memory benchmark

The existing memory benchmark was performed using the original N=3/K=3 configurations.

Conditions:

```text
Chunk size: 512
Batch size: 1
Datatype:   FP32
Optimizer:  Adam
```

For training, an initial optimizer step was performed before measurement so that Adam's optimizer-state buffers were allocated.

## Existing single-run measurements

| Model | Parameter memory | Inference peak allocated | Training peak allocated |
|---|---:|---:|---:|
| Baseline N=3 | 3.607 MiB | 25.61 MiB | 89.86 MiB |
| Looped K=3 | 1.213 MiB | 31.34 MiB | 82.66 MiB |
| Reinjected K=3 | 1.213 MiB | 31.59 MiB | 83.16 MiB |

Peak reserved memory:

| Model | Inference reserved | Training reserved |
|---|---:|---:|
| Baseline N=3 | 28.00 MiB | 138.00 MiB |
| Looped K=3 | 46.00 MiB | 136.00 MiB |
| Reinjected K=3 | 46.00 MiB | 136.00 MiB |

Parameter storage is reduced substantially by weight sharing.

However, total runtime memory does not fall proportionally because weights are only one component of the total footprint.

Training also requires:

```text
activations
GRU intermediates
gradients
optimizer state
CUDA workspaces
```

The looped models showed modestly lower peak training allocation but unexpectedly higher peak inference allocation.

These values currently come from one benchmark run and should not be treated as final efficiency results.

A repeated benchmark is still required.

---

# Updated conclusions

## 1. Recurrent weight sharing substantially reduces parameter count

At three effective depth passes:

```text
Baseline N=3:   945,602 parameters
Looped K=3:     317,890 parameters
Reinjected K=3: 317,891 parameters
```

This is approximately a 66.4% parameter reduction.

That result is deterministic.

---

## 2. Naive recurrent depth does not preserve performance

The ten-seed comparison shows:

```text
Baseline N=3:
0.55519 ± 0.01959

Looped K=3:
0.52185 ± 0.04252
```

Thus the original three-seed indication that shared recurrent depth preserved performance was misleading.

With more seeds, naive repeated weight tying clearly underperforms the independently parameterized baseline at K=3.

---

## 3. Recurrent-depth degradation increases with depth

The five-seed looped means are:

```text
K=1: 0.56652
K=2: 0.56149
K=3: 0.53871
K=4: 0.52567
```

The degradation is systematic across the tested recurrent depths.

---

## 4. Reinjection consistently mitigates recurrent-depth degradation

The paired reinjection improvement increases with depth:

```text
K=2: +0.00105
K=3: +0.00297
K=4: +0.01052
```

At K=2 and K=4, reinjection improved every one of the five paired seed comparisons.

This is currently the clearest positive result supporting the reinjection mechanism.

---

## 5. Reinjection is a mitigation mechanism, not a complete recovery

Even with reinjection:

```text
Reinjected K=4:
0.53620 ± 0.02161
```

remains substantially below:

```text
Baseline N=1:
0.56652 ± 0.00454

Baseline N=2:
0.56762 ± 0.00205
```

Therefore the current evidence does not support claiming that reinjection solves the recurrent-depth problem.

It partially mitigates it.

---

## 6. Shallow independent models are currently strongest

The strongest current mean is:

```text
Baseline N=2:
0.56762320 ± 0.00205026
```

The parameter-matched one-block model is close:

```text
Baseline N=1:
0.56652162 ± 0.00453565
```

This suggests that increasing architectural depth is not inherently beneficial for the present task and training regime.

---

## 7. Multi-seed evaluation materially changed the conclusion

The initial three-seed experiment suggested a positive result for recurrent weight sharing.

The ten-seed main comparison and five-seed depth study reversed that interpretation.

This demonstrates that conclusions about these models are sensitive to random initialization and should not be based on a very small number of seeds.

---

# Current working interpretation

The strongest claim presently supported by the data is:

> Repeated weight-tied depth in this dual-axis financial time-series architecture causes increasing predictive degradation as recurrent depth grows. Reinjecting the original encoded representation consistently mitigates this degradation, and the mitigation becomes larger at greater recurrent depth.

A second supported observation is:

> Under the present setup, recurrent computation does not outperform a shallow parameter-matched model, despite requiring repeated block execution.

The results do not currently support the stronger claim that recurrent depth provides a superior parameter/performance tradeoff.

---

# Remaining experiments

The experimental scope should now be narrowed rather than expanded.

## 1. Larger-data confirmation

The immediate next experiment should determine whether the current pattern survives when the model is trained on substantially more data.

Proposed screening setup:

```text
Training sequences:             1,000
Internal validation sequences:  200
Epochs:                         3
Initial seed:                   0
```

Representative configurations:

```text
Baseline N=1
Baseline N=2

Looped K=2
Reinjected K=2

Looped K=4
Reinjected K=4
```

This experiment tests:

1. whether N=1 and N=2 remain strong with more data;
2. whether repeated weight tying still degrades performance;
3. whether reinjection still improves the matched looped model;
4. whether the reinjection advantage still grows at larger recurrent depth.

If the qualitative pattern survives at seed 0, the key comparisons should then be repeated across additional seeds.

---

## 2. Final memory and efficiency characterization

The existing single-run memory benchmark should be repeated.

Final measurements should include:

```text
parameter count
parameter memory
peak training allocated memory
peak inference allocated memory
peak reserved memory
training runtime
inference runtime
```

Measurements should be repeated several times on the same GPU.

The final configurations used for memory benchmarking should be chosen after the larger-data screen so that efficiency is measured for the architectures actually discussed in the paper.

---

## 3. Reinjection-gate analysis

The learned reinjection gate should be summarized across:

```text
seed
recurrent depth
```

Useful questions include:

```text
Does the learned gate increase with recurrent depth?

Is a larger gate associated with larger recovery over plain looping?

How variable is the learned gate across seeds?
```

This provides a simple mechanistic analysis without adding a more complicated architecture.

---

## 4. Statistical analysis

Final comparisons should report:

```text
mean
standard deviation
paired seed differences
confidence intervals where appropriate
```

The reinjected-versus-looped comparison is naturally paired because corresponding runs use the same random seed and data split.

For the final selected configurations, paired statistical testing or bootstrap confidence intervals should be considered.

---

## 5. Optional fixed-gate ablation

If time permits, compare:

```text
plain looped
fixed reinjection gate
learned reinjection gate
```

This would help determine whether the improvement comes primarily from the existence of a source-reinjection path or from learning the amount of reinjection.

This experiment is lower priority than the larger-data confirmation.

---

## 6. External generalization

A second dataset would substantially strengthen the generality of the findings.

This is desirable for a broad machine-learning submission but is not required before determining whether the core Wunder result is robust.

---

# Short version for discussion

The larger experiment changed the story.

Naively reusing one Axial-GRU block does reduce parameters substantially, but repeated use of the same block progressively hurts prediction quality.

Across five seeds:

```text
Looped K=1: 0.5665
Looped K=2: 0.5615
Looped K=3: 0.5387
Looped K=4: 0.5257
```

Reinjecting the original encoded input does not completely solve this problem, but it consistently reduces the degradation:

```text
Reinjection advantage:

K=2: +0.0010
K=3: +0.0030
K=4: +0.0105
```

The benefit of reinjection therefore grows as recurrent depth becomes more damaging.

The strongest model currently is a conventional two-block independent baseline:

```text
Baseline N=2:
0.5676 ± 0.0021
```

A one-block model is nearly as strong:

```text
Baseline N=1:
0.5665 ± 0.0045
```

The current contribution is therefore not that recurrent depth beats a shallow parameter-matched model.

The more defensible result is that recurrent weight tying introduces depth-dependent degradation, and source reinjection provides a simple, low-parameter mechanism that consistently mitigates that degradation.

The next critical test is whether this pattern survives when training is expanded from 100 to 1,000 sequences.
