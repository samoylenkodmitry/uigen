# V6 Conclusions for GPT-5.5 Pro

Date: 2026-05-23

## Executive Summary

V6 is a real architectural improvement, but the current V6 copy/refine
architecture is not enough to scale.

It proves the important new mechanism:

```text
rendered input pixel evidence -> predicted BMP-pixel UV correspondence
```

But it does not yet meet the full visible-copy quality target.

Best current checkpoint:

```text
runs/slotnet_v6_default_skin_stage2_p4residual/snapshot_step005000.safetensors
```

Best metrics:

```text
uv_median_px   0.63    target < 2.0    PASS
copy_conf_auc  0.9707  target > 0.98   near miss
visible_mae    0.0185  target < 0.01   PARTIAL / FAIL
```

Short verdict:

```text
UV correspondence: proven
source-preserving copy quality: not solved
residual correction: small useful gain, not enough
do not scale / rent GPU for current V6 as-is
```

## What V6 Fixed

V6 fixed the main V5 problem. V5 could identify skin identity but generated
pixels from compressed features, so photographic/high-frequency skins blurred.

V6 uses final-frame Cranamp provenance labels and trains the model to predict
where each exported BMP pixel appears in the rendered input.

This part works.

Evidence:

```text
initial normalized UV-loss run:
  uv_median_px ~383 px at 5k

pixel-space UV-first training:
  uv_median_px 6.86 at 5k
  uv_median_px 4.25 at 10k
  uv_median_px 2.88 at 20k

subpixel refinement:
  uv_median_px 0.66 at p3 2k

best residual checkpoint:
  uv_median_px 0.63
```

So the provenance labels, dataset, model conditioning, and UV loss are
capable of learning BMP-pixel -> input-pixel correspondence.

## Final Stage 2 State

The best checkpoint remains p4 residual step 5k:

```text
runs/slotnet_v6_default_skin_stage2_p4residual/snapshot_step005000.safetensors
```

Aggregate:

```text
visible_mae    0.0185
copy_conf_auc  0.9707
uv_median_px   0.633
```

Per-file visible MAE at p4 5k:

```text
file        visible_mae   AUC      UV px
EQMAIN      0.0281        0.9134   0.873
BALANCE     0.0270        0.8593   0.490
CBUTTONS    0.0195        0.9279   0.639
MONOSTER    0.0157        0.9994   0.347
PLEDIT      0.0145        0.9988   0.663
TITLEBAR    0.0139        1.0000   0.369
VOLUME      0.0108        0.8715   0.540
SHUFREP     0.0106        0.9545   0.602
MAIN        0.0099        0.9751   0.482
POSBAR      0.0062        0.8073   0.738
PLAYPAUS    0.0006        1.0000   0.162
```

Three files clear the strict 0.01 visible-MAE target at this checkpoint.
The pattern is stable: EQMAIN/BALANCE/CBUTTONS and other dense sprite files
still set the floor.

## Stage 2 Training Trajectory

Main trajectory:

```text
point        visible_mae   AUC     UV px
p1  5k        0.1126      0.8588   6.86
p1 10k        0.0902      0.9141   4.25
p1 20k        0.0695      0.9522   2.88
p1+25k        0.0726      0.9599   2.14
p1+28k        0.0582      0.9632   2.17   clean phase-1 checkpoint
p1+30k        0.0702      0.9652   2.43   EQMAIN drifted
p2  2k        0.0377      0.9682   1.27
p2 10k        0.0470      0.9785   1.48
p3  2k        0.0251      0.9678   0.66   best UV-only/subpixel copy
p4  5k        0.0185      0.9707   0.63   best overall
```

Residual probes:

```text
p4 joint residual_l1=0.02:
  visible_mae 0.0251 -> 0.0185
  UV held subpixel
  useful, but not enough

p5 joint residual_l1=0.002:
  visible_mae stuck/worse at 0.021-0.023
  looser residual did not help

p6 freeze backbone, train residual only:
  visible_mae 0.0185 -> 0.0186
  residual head alone is effectively impotent
```

Conclusion from residual tests:

```text
The p4 gain came from joint backbone/head adjustment, not from the residual
head independently solving copy error.

Small additive residual is useful in a narrow range, but it does not close
the remaining copy-quality gap.
```

## Diagnostics

### Bilinear vs Nearest

At p3 2k:

```text
bilinear visible_mae  0.0251
nearest visible_mae   0.0212
```

Nearest improves the metric, so bilinear blur is part of the gap. But the
gap remains far above 0.01, so the issue is not only bilinear interpolation.

### UV Tail

At p3 2k, median UV is excellent, but EQMAIN has a tail:

```text
EQMAIN UV error:
p50   0.98 px
p75   1.63 px
p90   3.15 px
p95   4.80 px
p99  11.76 px
```

This suggests median UV is no longer enough as the acceptance metric. The
next metrics should probably include UV p90/p95 and per-file thresholds.

### Confidence

AUC improved but stayed under target:

```text
best AUC 0.9707
target   0.98
```

This is not the main blocker for Stage 2 because UV was the core unknown,
but low-AUC files such as BALANCE/VOLUME/POSBAR show the confidence head
also needs attention before final copy/fallback composition.

## Current Interpretation

The current V6 architecture works for:

```text
correspondence learning
subpixel median UV
source-aware copy direction
```

It does not work for:

```text
strict visible-pixel reconstruction
high-frequency source preservation under hard files
closing residual RGB mismatch with a tiny additive residual
```

The remaining error likely lives in some combination of:

```text
1. UV tail / local misregistration, especially EQMAIN and hard dense files
2. copy mechanism limits: bilinear/nearest sampling from a distorted render
3. insufficient per-file local capacity for hard files
4. missing fallback/completer branch for pixels where copy evidence is weak
5. copy_conf not sharp enough for final copy-vs-fallback gating
```

## Recommended Decision Point

Do not continue training current V6 recipes as-is.

Do not scale to many skins yet.

Ask GPT-5.5 Pro to choose the next architectural step:

```text
A. Add fallback/completer branch now and train final composition
B. Increase per-file capacity / local conditioning for hard files first
C. Change copy mechanism: nearest/quantized UV, local patch sampler, or UV-tail loss
D. Revise Stage 2 acceptance: median UV pass may be sufficient, visible_mae should
   be owned by Stage 3 fallback/completer
```

My bias:

```text
Stop pure Stage 2 tuning.
Keep p4 as the V6 correspondence baseline.
Next useful experiment should either:
  - add fallback/completer composition, or
  - directly attack UV tail / hard-file capacity.

Do not spend more time on looser residual penalties.
```

## Files and Commits

Relevant committed infrastructure:

```text
077f24a Add final-frame provenance buffer to cranamp renderer
6543bbe Fix scaled provenance source mapping
85f1387 Add V6 provenance label generation
1bbf429 Add V6 dataset generator
e312bba Add V6 copy-stage training path
b9a7824 Add pixel-space V6 UV training controls
606d18d Report V6 Stage 2: UV correspondence passes, copy quality partial
b91caf1 Add V6 copy_residual head + probe support
cfae482 Add V6 residual-only freeze probe
```

Local generated dataset remains untracked:

```text
data_v6_default_skin_stage2/
```
