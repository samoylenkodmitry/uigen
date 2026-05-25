# V7 Phase 0 Report For GPT-5.5 Pro

Date: 2026-05-25

## Short Version

We cleaned up V7 Phase 0 and got a decisive result:

```text
The current V7Completer is not sufficient as a hidden-region generator.
It learns low-frequency structure but fails pixel-crisp high-frequency detail,
even in a one-file / one-skin / one-image overfit.
```

The strongest evidence is the strict MAIN floor test:

```text
skin:     a_halo_so_bright_it_bleeds
file:     MAIN.bmp only
mask:     random_rect=1.0
model:    V7Completer c48
steps:    30k
metric:   hidden-normalized, not support-normalized

final hidden_mae:  0.0246
final hidden_hit5: 0.455
target:            hidden_mae < 0.015 and hidden_hit5 > 0.90
```

This is the cheapest controlled case. Since it fails there, we should not
launch another 14-skin / c64 / long-Kaggle run with the same U-Net recipe.

## Current Project Goal

Final product target remains:

```text
input mockup/render image -> exported Winamp/Cranamp BMP tensors -> .wsz
```

V7 Phase 0 only tests the internal completer:

```text
partial exported BMP evidence + observed mask + file/skin conditioning
-> completed exported BMP tensor
```

This is not the full deployed model. It is the hidden-state completion branch
we need before reconnecting observer/copy.

## What Was Fixed Before This Result

Several earlier V7 results were contaminated. These are now fixed in `main`.

### 1. Hidden-Normalized Loss And Metrics

Old V7 metrics were support-normalized:

```python
loss = sum(abs(final - target) * support) / sum(support)
```

But the completer hard-copies observed pixels:

```python
final = observed_mask * observed_rgb + (1 - observed_mask) * generated
```

So old metrics were diluted by copied observed pixels.

New primary metrics use:

```python
hidden = (1 - observed_mask) * support
```

and report:

```text
hidden_supported_mae
hidden_hit5
hidden_sobel_mae
observed_passthrough_mae
```

`observed_passthrough_mae` is 0.0 in the re-evals and floor test, so the
copy path is exact and the hidden denominators are trustworthy.

### 2. State-Family Semantics

The state-family metadata now has:

```yaml
mask_role: alternatives | components | single
```

Only `alternatives` families are eligible for reveal-one-hide-siblings masks.

Important correction:

```text
POSBAR track and thumb are components, not sibling alternatives.
```

So POSBAR is no longer trained/evaluated as "reconstruct track from only thumb"
or "reconstruct thumb from only track".

### 3. Eval Coverage And Per-Mode Metrics

Eval now reports:

```text
coverage.evaluated_files
coverage.skipped_files
per_mode hidden metrics
per_file hidden metrics
per_skin hidden metrics
```

Pure `state_family` eval skips component-only files instead of crashing.

### 4. Trainer Guard

The trainer now fail-fasts if a sampled file has no eligible mask mode under
the requested mask mix. This prevents mid-run crashes from state_family-only
training over component-only files.

## Corrected Re-Eval Of Old Phase B Checkpoint

Checkpoint:

```text
runs/gateB_curriculum_phaseB/best.safetensors
```

This checkpoint was trained under the old support-normalized loss. We re-evaled
it under corrected hidden metrics.

### Aggregate Results

| eval | hidden_mae | hidden_hit5 | old full_mae | old full_hit5 | obs_pass |
|---|---:|---:|---:|---:|---:|
| A: state_family-only, 7 files | 0.0539 | 0.608 | 0.0228 | 0.834 | 0.0000 |
| B: sf0.7 / random_rect0.3, 11 files | 0.0654 | 0.570 | 0.0140 | 0.908 | 0.0000 |

Conclusion:

```text
The old metrics were misleading by roughly 3-4x.
The model was never close to Gate B.
```

### Per-Mode Finding

Corrected mixed eval:

| mode | hidden_mae | hidden_hit5 |
|---|---:|---:|
| state_family | 0.0484 | 0.618 |
| random_rect | 0.1983 | 0.196 |

The surprise is `random_rect`. We previously thought it was trivial because
support-normalized metrics hid the error behind the copied pixels. With hidden
metrics, arbitrary-region inpainting is the worst mode by far.

### Per-File Hidden Metrics In Mixed Eval

| file | hidden_mae | hidden_hit5 |
|---|---:|---:|
| VOLUME | 0.0164 | 0.757 |
| BALANCE | 0.0245 | 0.565 |
| PLAYPAUS | 0.0380 | 0.700 |
| MONOSTER | 0.0537 | 0.413 |
| SHUFREP | 0.1454 | 0.225 |
| POSBAR | 0.1578 | 0.244 |
| CBUTTONS | 0.1735 | 0.149 |
| EQMAIN | 0.1871 | 0.297 |
| MAIN | 0.2192 | 0.104 |
| PLEDIT | 0.2474 | 0.148 |
| TITLEBAR | 0.3701 | 0.042 |

The curriculum-emphasized strip files were best, but still not enough.
Component-only files under random-rect are catastrophic.

## Sanity Probes After Cleanup

These proved the corrected loss can train, but not to the required floor.

### Probe A: One Skin, MAIN Only, Random-Rect

```text
MAIN hidden_l1 0.0996 -> 0.036 over 8k
```

Monotonic, so the loss works and the model learns something.

### Probe B: One Skin, All Files, Random-Rect Heavy

```text
overall hidden_l1 0.0997 -> 0.0501 over 6k
every file improved
```

### Probe C: 14 Skins, Oracle Skin ID, Random-Rect Heavy

```text
overall hidden_l1 0.2368 -> 0.1465 over 4k
every file improved
```

These probes show the code path is not broken. But they do not prove the model
can reach the pixel-crisp gate.

## Decisive Floor Test

We then pre-registered a strict floor test:

```text
If one MAIN image cannot overfit under random_rect,
do not launch Gate B and do not scale the same recipe.
```

Setup:

```text
skin:   a_halo_so_bright_it_bleeds
file:   MAIN.bmp
mask:   random_rect=1.0
model:  V7Completer c48
loss:   hidden-normalized + sobel 0.25
lr:     1e-3
batch:  1
steps:  30k
eval:   mask_samples=16
```

Results:

| snapshot | hidden_mae | hidden_hit5 | obs_passthrough |
|---:|---:|---:|---:|
| 5k | 0.2063 | 0.221 | 0.0 |
| 10k | 0.0982 | 0.282 | 0.0 |
| 15k | 0.0509 | 0.325 | 0.0 |
| 20k | 0.0405 | 0.370 | 0.0 |
| 25k | 0.0298 | 0.409 | 0.0 |
| 30k | 0.0246 | 0.455 | 0.0 |

Acceptance:

```text
hidden_mae < 0.015
hidden_hit5 > 0.90
```

Neither passed.

The key number is hit5:

```text
After 30k steps on one image, fewer than half the hidden pixels are within
5/255. The curve is still moving but decelerating. It is not plausibly heading
to 0.90 with this architecture.
```

## Current Interpretation

The current V7Completer:

```text
observed_rgb + observed_mask + file_id + skin_id + Fourier coords
-> small masked U-Net
-> generated hidden pixels
-> hard-copy observed pixels
```

is a useful diagnostic baseline, but not an adequate hidden-region generator.

It captures low-frequency structure. It does not reconstruct crisp edges,
texture, glyphs, and high-frequency pixel details well enough. The error is
bimodal: flat regions become near-perfect, but detail/edge pixels remain wrong,
so MAE can look tolerable while hit5 collapses.

This is now unlikely to be:

```text
metric bug
POSBAR mask bug
sampler bug
curriculum bug
data budget issue
14-skin difficulty
```

because the strict one-image MAIN floor test already fails.

## Architecture Question

We need a deliberate architecture decision, not another long run.

Candidate directions:

### 1. Stronger Per-Pixel Coordinate Representation

More Fourier bands, learned coordinate embeddings, or per-file coordinate
tables. Hypothesis: the current conditioning cannot address sharp features
precisely enough.

Concern: pure coordinate generation may still produce smooth approximations.

### 2. Final-Resolution Refinement Stage

Keep the current completer as a coarse prediction, then add a refinement head
operating at full BMP resolution with local residual blocks and edge-focused
loss. Hypothesis: the current decoder is too smooth at output resolution.

Concern: if the missing texture cannot be inferred from input evidence, a
refiner may only sharpen hallucinations.

### 3. Copy / Patch / Retrieval Path From Observed Pixels

For hidden random-rect regions, use observed pixels from the same BMP/skin as
source evidence:

```text
observed pixels + mask -> patch/texture retrieval/refinement -> hidden pixels
```

This may be a learned patch sampler, attention over observed BMP pixels, or a
V6-style copy/refine mechanism inside the completer.

Hypothesis: visible details should be preserved/copied/reused, not generated
from compressed style and coords.

Concern: must still handle truly hidden state siblings, so it needs a fallback
generator too.

### 4. File-Group-Specific Decoders

Separate heads for:

```text
large panels: MAIN, EQMAIN, PLEDIT
state strips: VOLUME, BALANCE, EQMAIN slider frames
small sprites: CBUTTONS, SHUFREP, MONOSTER, PLAYPAUS, POSBAR, TITLEBAR
```

Hypothesis: one shared U-Net is forcing too many geometries into one decoder.

Concern: the MAIN floor test is one file only, so file grouping alone probably
does not solve the high-frequency problem.

## What We Need From GPT-5.5 Pro

Please reason from the evidence above and recommend the next minimal
architecture experiment.

Important constraints:

```text
Do not suggest another long c48/c64 run of the same V7Completer.
Do not rely on support-normalized metrics.
Do not reintroduce default priors, full-atlas loss, unsupported files, or
distortion metadata.
Final product still must be one model/inference path:
  input image -> exported BMPs -> .wsz
```

Question:

```text
What should replace or augment the current masked U-Net completer so hidden
regions can be reconstructed with pixel-crisp detail?

What is the cheapest experiment that would falsify or validate that direction?
```

My current bias is:

```text
Use a copy/patch/retrieval path from observed pixels plus a final-resolution
refinement stage. Pure generation from file/skin embeddings and coordinates
has now failed the strict floor test.
```
