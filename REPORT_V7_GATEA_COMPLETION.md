# V7 Phase 0 / Gate A — One-Skin Asset Completion

Date: 2026-05-23

## Verdict: PASS

All four user-specified acceptance criteria are met at the 55k checkpoint
on default_skin:

```text
aggregate supported_mae      0.0029   target < 0.005    PASS
aggregate hit5               0.960    target > 0.95    PASS
every per-file sup_mae <0.005  11/11                    PASS
no per-file hit5 < 0.90        0 files                  PASS
files with hit5 > 0.95         8/11   ideal: all        partial
```

## Best Checkpoint

```text
runs/v7_completer_gateA_weighted_c48_sobel/snapshot_step005000.safetensors
```

This is the V7 completer model after:

```text
phase 1   round-robin sampling, c48, 5k     steps (Gate A first attempt)
phase 2   c24 5k vs c48 5k baseline runs (diagnostics)
probes    all-observed / whole-file / state-family / EQMAIN-only
phase 3   weighted sampling, c48, 30k       steps (Gate A 2nd attempt)
phase 4   weighted resume, weight bumps, +20k -> absolute 50k
phase 5   weighted resume + sobel 0.25, +5k -> absolute 55k         (accepted)
phase 5+  ran to +10k -> absolute 60k; regressed; rejected
```

The selection is by aggregate eval, not by trainer best-batch-loss. The
60k snapshot is rejected because aggregate hit5 fell back to 0.934 and
VOLUME specifically dropped from 0.971 to 0.816 — likely AdamW
oscillation post-sobel introduction. Not investigated further because
Gate A already passes at 55k.

## Final Per-File Metrics

```text
file            sup_mae    hit5     sobel_mae   notes
MAIN.bmp        0.0005     0.989    0.0024      passes both gates
TITLEBAR.bmp    0.0010     0.983    0.0080      passes both gates
SHUFREP.bmp     0.0016     0.975    0.0262      passes both gates
PLEDIT.bmp      0.0019     0.959    0.0282      passes both gates
PLAYPAUS.bmp    0.0023     0.911    0.0174      hit5 below 0.95
CBUTTONS.bmp    0.0026     0.952    0.0167      passes both gates
VOLUME.bmp      0.0030     0.971    0.2734      passes both gates *
MONOSTER.bmp    0.0031     0.932    0.0141      hit5 below 0.95
POSBAR.bmp      0.0034     0.993    0.0110      passes both gates
BALANCE.bmp     0.0037     0.959    0.2018      passes both gates
EQMAIN.bmp      0.0042     0.939    0.0333      hit5 below 0.95
```

`*` VOLUME's hit5 0.971 is post-sobel; it was 0.83 immediately before
sobel was introduced.

## Diagnosis Trail

Gate A took five training attempts and four diagnostic probes. The
sequence of failures and fixes is the load-bearing part of the report.

### Attempt 1: default plan + c24, 5k

```text
sup_mae   0.041
hit5      0.32
```

Far from target. Trajectory still descending. Bumped capacity and
training length.

### Attempt 2: c48, 15k

```text
sup_mae   0.024 (-41% vs attempt 1)
hit5      0.67
```

Better but not passing. Per-file: 2/11 pass strict gates; EQMAIN stuck
at 0.037, MAIN/TITLEBAR/SHUFREP/CBUTTONS all 0.02-0.05. Decided to
diagnose before throwing more steps at the problem.

### Probe 1: all-observed copy-through

Trained with `--mask-passthrough 1.0` so observed_mask = support
everywhere. The model gets the target as input inside the support
region; a working completer should output the target essentially
unchanged.

```text
step 200:   sup_mae 0.090   hit5 0.10
step 2000:  sup_mae 0.022   hit5 0.38
```

This is the smoking gun: with the target literally as input, the
completer could not pass it through accurately. Diagnosis: the U-Net
had no input->output skip, so the model was using all its capacity
just to learn the identity mapping through encoder+decoder. The fix
is structural, not architectural-bigger.

### Fix: hard observed-mask copy

```python
generated_rgb = sigmoid(rgb_logits)
final_rgb     = observed_mask * observed_rgb + (1 - observed_mask) * generated_rgb
```

Known pixels are known by construction. The completer's loss only
sees the hidden region. The new tests assert that
`support_masked_l1_loss(final, target, support) == 0` exactly, with no
training, on an all-observed sample.

Probe 1 re-ran: sup_mae = 0.00000000 at every step, every file. The
fix is exact and analytically validated.

### Probe 2: whole-file memorization (mask=0 everywhere)

```text
step 1k:    sup_mae 0.120   hit5 0.11
step 10k:   sup_mae 0.027   hit5 0.57
```

Generated branch slowly learns to memorize files from `file_id` +
Fourier coords alone, but EQMAIN/PLEDIT/MAIN stuck high. With c48
+15k extended: aggregate 0.019 / 0.68. Still failing on EQMAIN
specifically (0.032 / 0.49).

### Probe 3: state-family-only

c48 + 15k, weights forcing state_family mode:

```text
aggregate sup_mae   0.0185 (similar to whole-file 0.019)
aggregate hit5      0.849  (much better than whole-file 0.68 because
                            hard-copy reveals lots of observed pixels)
```

Per-file: most files improved (more observed -> easier), but EQMAIN
got worse (state_family masks hide 27 of 28 slider frames in a
structured pattern, hardest mode for that file). MONOSTER also worse
(2-state strip, hides half the BMP).

EQMAIN was uniquely bad in both probes. Per the diagnosis tree, this
called for an EQMAIN-only overfit probe to separate "model can't do
EQMAIN" from "EQMAIN gets starved in multi-file training".

### Probe 4: EQMAIN-only, two mask modes

Filtered the dataset to only EQMAIN.bmp items and trained c48 + 15k
under whole_file and state_family masks separately:

```text
whole_file    @15k:  sup_mae 0.00081   hit5 0.9996   PASS
state_family  @15k:  sup_mae 0.00240   hit5 0.981    PASS
```

Both pass cleanly. The model can memorize EQMAIN in isolation under
both mask modes. The multi-file failure is **gradient starvation**:
with batch=1 and 11 files in round-robin, EQMAIN gets 1/11 of the
optimizer steps. 15k/11 ~ 1300 EQMAIN-effective steps is below the
~5k it needs to converge.

### Fix: weighted same-file sampling

`WeightedSameFileBatchSampler` samples each step's file from a
configurable probability distribution, with replacement within the
file group (necessary for one-skin where each file has exactly one
item). The default weights bias the gradient budget toward harder
files:

```text
EQMAIN.bmp    8
MAIN.bmp      4
TITLEBAR.bmp  4
CBUTTONS.bmp  4
SHUFREP.bmp   4
PLEDIT.bmp    4
VOLUME.bmp    3
BALANCE.bmp   3
MONOSTER.bmp  2
POSBAR.bmp    1
PLAYPAUS.bmp  1
```

### Attempt 3: c48 weighted, 30k

```text
aggregate sup_mae   0.0047    PASS (<0.005)
aggregate hit5      0.933     FAIL (>0.95)
5/11 files pass both gates
```

Aggregate MAE crosses the bar for the first time. EQMAIN moved from
0.04 to 0.007. New bottleneck: MONOSTER (0.015, hit5 0.75) and
CBUTTONS (0.009, hit5 0.84). Both had been under-weighted relative to
their difficulty.

### Attempt 4: weighted resume + reweight, +20k -> 50k

`configs/v7_file_weights_continuation.yaml` bumps MONOSTER 2->8,
CBUTTONS 4->6, VOLUME 3->5, PLAYPAUS 1->2, BALANCE 3->4. Resumed from
the 30k checkpoint.

```text
@50k absolute:
  aggregate sup_mae   0.0036    PASS
  aggregate hit5      0.936     FAIL (still <0.95)
  8/11 files pass both gates
  remaining bottleneck: VOLUME at sup_mae 0.0050 / hit5 0.83
```

VOLUME's MAE was effectively at the bar but hit5 stuck at 0.83.
Visual diagnosis: predictions looked correct on the contact sheet,
but with consistent 5-6 level color error on slider highlight bars -
classic edge-precision failure. The hit5 threshold of 5/255 catches
exactly this kind of "almost right" pixel.

### Fix: support_masked_sobel_mae in training loss

`--sobel-weight` CLI added. With sobel_weight=0.25 the training total
becomes `l1 + 0.25 * support_masked_sobel_mae`.

### Attempt 5: weighted resume + sobel 0.25, +5k -> 55k

```text
@52k:    sup_mae 0.0033   hit5 0.951   sobel 0.083
@55k:    sup_mae 0.0029   hit5 0.960   sobel 0.076    (accepted)
@60k:    sup_mae 0.0033   hit5 0.934   sobel 0.077    (rejected, oscillation)
```

VOLUME hit5 jumped 0.83 -> 0.97 in this window. sobel_mae dropped 3x.
55k is the clean stop point.

## What This Locks In

Three landed changes that should carry forward to Gate B and beyond:

1. **Hard observed-mask copy contract** in `V7Completer.forward`:
   `known_pixels = observed_rgb`, `hidden_pixels = sigmoid(rgb_logits)`.
   Not optional. The completer should never spend capacity learning
   identity on pixels it was already given.

2. **Weighted same-file sampling**: round-robin is wrong when file
   difficulty varies. The weights are an empirical tuning knob;
   defaults match what worked for Gate A and should be re-validated
   for Gate B's 16-skin set.

3. **Sobel edge precision** is a real and necessary signal for files
   like VOLUME / BALANCE whose support is mostly thin color bands.
   Pure L1 plateaus at "visually correct but 5-6 level off" exactly
   where hit5 cares about precision.

## Cost

```text
total wall time for Gate A     ~25 minutes on RTX 2070
trainer steps                  55000 (one-skin)
diagnostic probes              4 (Probes 1-4 across ~25k extra steps)
final V7 completer params      ~1.0M (c48, file_emb=32, frequencies=(1,2,4,8))
```

## Limitations / Watch Items for Gate B

```text
- The accepted checkpoint trains only default_skin. Whether the same
  recipe generalizes to 16 skins is a hypothesis, not a result. Gate B
  must validate it.

- AdamW oscillation post-sobel: 55k passed cleanly, 60k regressed.
  For Gate B, watch the eval trajectory and stop when aggregate hit5
  crosses the bar; do not let "more steps" silently damage VOLUME.

- 3 files have hit5 in [0.91, 0.95]: PLAYPAUS, MONOSTER, EQMAIN. They
  pass the floor but not the "ideal" gate. May need more state-mask
  curriculum or per-file weighting on the 16-skin run.

- The completer's edge-precision capacity may not generalize to
  unseen photographic skins (TVXQ from V5 Gate C, etc.). Gate E (hard-
  skin mini-gate) will surface this.
```

## Next: Gate B Plan (Not Yet Started)

Gate B target per the V7 plan:

```text
16-skin asset completion
  retrieval top1 = 1.0     (if a retrieval eval exists for completer outputs)
  median MAE     < 0.015
  hit5           > 0.90
```

Recipe to validate:

```text
- same V7Completer with hard-mask copy
- WeightedSameFileBatchSampler, default weights (re-validate)
- --base-channels 48
- --sobel-weight 0.25
- support-masked eval
- best checkpoint selected by aggregate eval, not last step
- early stop if hit5 oscillates / regresses after passing
- train from scratch, not warm-started from default_skin
  (warm-start would bias completion toward default-skin layout;
   Phase 0 is cleaner from scratch).
```

To be decided before Gate B launches:

```text
1. 16 skin sources (use the same data_v4_16skin set V5/V6 used).
2. Step budget. One-skin Gate A used 55k. With 16 skins, each file
   still gets weighted exposure; weighted sampling cares about per-
   file gradient steps, not per-skin, so 55k may be enough or may
   need 2-3x more.
3. Whether to add a retrieval eval for the completer (it does not
   produce a global style vector; revisit).
```

Architecture and loss are unchanged.
