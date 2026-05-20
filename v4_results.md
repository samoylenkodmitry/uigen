# V4 Results Summary

This document summarizes the current state of the Cranamp skin model after
V3.5/V4. It is written as an external review note: what changed, what passed,
what failed, and what the next architecture should address.

## Current Training Contract

The project is no longer training against a padded full atlas.

The active contract is:

```text
input render PNG -> predicted exported BMP tensors -> expected BMP pixels
```

Important constraints now in the code:

- No prior/default atlas.
- No distortion metadata as input.
- No dynamic mask as model input.
- No full-atlas metric as pass/fail.
- Loss/eval use exact exported BMP tensors.
- Loss/eval score only static Cranamp-supported pixels.
- Unsupported BMPs such as `VIDEO.bmp`, `TEXT.bmp`, and `NUMBERS.bmp` do not
  participate in training.

This was the first setup that produced meaningful one-skin overfits.

## V3.5 / V4 Positive Result

V3.5 changed the target from a huge padded atlas to exact exported BMP tensors.
V4 added static Cranamp-supported-pixel masking, AMP, workers, retrieval eval,
and reproducible reports.

Gate 2 tested diverse one-skin overfits with masked loss.

| skin | steps | exported MAE | hit5 | EQMAIN MAE | EQMAIN hit5 | verdict |
|------|------:|-------------:|-----:|-----------:|------------:|---------|
| darkside | 10k | 0.00431 | 0.971 | 0.0088 | 0.940 | clean pass |
| Aguileramp | 20k | 0.00577 | 0.960 | 0.0141 | 0.845 | pass, EQMAIN at threshold |
| Zelda | 20k | 0.00421 | 0.955 | 0.0146 | 0.754 | aggregate pass, EQMAIN-hit5 caveat |

Conclusion from Gate 2:

```text
The model/loss/export path can overfit one skin sharply enough to be useful.
The corrected target and masked loss are valid.
EQMAIN hit5 is a recurring weak point.
```

## Gate 3 Setup

Gate 3 tested whether V4 can memorize 16 skins at once.

Dataset:

```text
data_v4_16skin/train.csv
16 skins * 32 variants = 512 rows
train-only memorization split
```

Run:

```text
run dir: runs/slotnet_v4_16skin_masked_mem
checkpoint used: snapshot_step050000.safetensors
git_commit recorded in config: 595bd09
steps: 50000
batch: 2
base_channels: 24
style_dim: 192
edge_weight: 1.5
AMP: enabled
```

The 16 skins were selected by diversity over brightness, contrast,
saturation, palette estimate, and edge density. The set included the three
Gate 2 anchors plus 13 diverse skins:

```text
darkside
Aguileramp_-_OldSchool
Zelda_Amp_Gold
GoodGawd
The_Four_Horsemen
engraved4_platinum
minimalistic_black
a_halo_so_bright_it_bleeds
Cyborg
Rancid_Amp_5
tvxq_winamp_skins_by_roseweedy
simblyblayit
Infected FX - Gray No Transparency
Ruki2 by michi
DragonZV30amp
blair_razor_project
```

## Gate 3 Final Result

Gate 3 is a useful failure.

V4 separates skin identities perfectly, but it does not reconstruct skin
texture/detail well enough.

### Aggregate Metrics

Full-dataset eval on `snapshot_step050000.safetensors`:

| metric | result |
|--------|-------:|
| samples | 512 |
| retrieval top1 | 1.000 |
| mean true exported MAE | 0.04065 |
| median true exported MAE | 0.03879 |
| exported_pixels_mae | 0.04065 |
| exported_pixels_hit_5_255 | 0.668 |
| exported_pixels_sobel_mae | 0.05852 |

Acceptance target for Gate 3 was:

```text
top1 retrieval accuracy > 95%
median exported_pixels_mae < 0.02
no identity collapse
```

Result:

```text
retrieval passes cleanly
reconstruction fails clearly
```

### Per-File Metrics

Sorted by MAE:

| file | MAE | hit5 | sobel MAE |
|------|----:|-----:|----------:|
| EQMAIN | 0.07830 | 0.567 | 0.11375 |
| VOLUME | 0.05936 | 0.517 | 0.07332 |
| MAIN | 0.05675 | 0.619 | 0.08911 |
| CBUTTONS | 0.05161 | 0.578 | 0.08405 |
| BALANCE | 0.05002 | 0.547 | 0.06101 |
| PLEDIT | 0.05000 | 0.644 | 0.08666 |
| SHUFREP | 0.03192 | 0.597 | 0.04648 |
| POSBAR | 0.03139 | 0.663 | 0.03452 |
| TITLEBAR | 0.02642 | 0.738 | 0.04271 |
| MONOSTER | 0.00911 | 0.887 | 0.00906 |
| PLAYPAUS | 0.00222 | 0.995 | 0.00307 |

The small files are solved or near-solved. Larger and texture-heavy files are
not.

## Interpretation

The result is diagnostic:

```text
V4 learns which skin it is looking at.
V4 cannot reconstruct enough skin-specific texture/detail through one global
192-dimensional style vector.
```

Retrieval top1 is 1.000, so this is not identity collapse. The model picks the
correct target skin for every input variant.

The reconstruction failure means the global style vector is a bottleneck. It
contains enough information to separate skins but not enough local evidence to
rebuild large BMPs like `EQMAIN`, `MAIN`, `VOLUME`, `PLEDIT`, `CBUTTONS`, and
`BALANCE`.

This also explains why one-skin overfit passed: with one identity, the decoder
can memorize the skin. With 16 identities, the shared global vector and file
heads are forced to encode too much skin-specific texture into too small and too
global a representation.

## What Not To Do Next

Do not continue the V4 schedule.

Do not rent a bigger GPU for this architecture yet.

Do not return to padded full-atlas training.

Do not try to solve this primarily with file weights or longer schedules. V4's
training curve was still moving but decelerating; extrapolation did not point
toward the `< 0.02` Gate 3 reconstruction target.

Do not reintroduce priors, distortion metadata, or dynamic masks as model
inputs.

## Recommended V5 Direction

Keep the parts that worked:

- input render -> exact exported BMP tensors
- static Cranamp-supported-pixel loss/eval
- per-file output tensors
- retrieval eval
- no prior atlas
- no distortion metadata
- no full-atlas pass/fail metric

Change the conditioning path:

```text
V4:
  encoder -> global style vector -> per-file decoders

V5:
  encoder -> spatial feature maps
          -> per-file learned queries / cross-attention
          -> per-file decoders
```

Each file head should be able to pull local evidence from the rendered input,
not only a compressed global style vector.

Suggested V5 structure:

```text
shared CNN encoder:
  keep spatial feature map, not only pooled style

global style:
  keep as auxiliary conditioning

per-file head:
  learned file/query token
  cross-attend into encoder spatial features with 2D positional encoding
  decode exact BMP tensor using:
    - attended local context
    - global style vector
    - file embedding
    - Fourier x/y coordinate maps
```

Run order for V5:

```text
1. unit tests for output shapes and supported-pixel gradients
2. one-skin overfit sanity
3. three-skin quick overfit
4. repeat 16-skin Gate 3
```

V5 should be judged by the same metrics:

```text
retrieval top1
median true exported_pixels_mae
per-file MAE / hit5 / sobel
visual renders / exported skin loadability
```

## Bottom Line

V4 is a real step forward because it proves the corrected data path works and
that the input contains enough signal to separate skin identities.

It also proves the current architecture is not enough for multi-skin texture
reconstruction.

The next useful move is not more V4 training. It is a V5 local-evidence
architecture.
