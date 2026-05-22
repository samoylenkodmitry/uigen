# SlotNet V5 Full Review

Date: 2026-05-22

## Executive Summary

V5 is a real improvement over V4, but it is not a final source-preserving
architecture.

What V5 fixed:

```text
- exact exported BMP tensor target instead of padded atlas target
- static Cranamp-supported-pixel loss/eval
- no default prior atlas
- no distortion metadata in the model path
- no dynamic masks as model inputs
- per-file metrics and retrieval eval
- local encoder feature access through per-file cross-attention
```

What V5 proved:

```text
- one-skin overfit works very well
- three-skin overfit works very well
- sixteen-skin identity retrieval is perfect
- sixteen-skin reconstruction is much better than V4
```

What V5 did not prove:

```text
- strict high-frequency pixel preservation
- robust reconstruction of photographic / saturated skins
- reliable per-file localization from attention maps
- production readiness for broad unseen GPT/Gemini-style mockups
```

The key Gate C result:

```text
V4 Gate 3 @50k:  MAE 0.04065, hit5 0.668, retrieval 1.000
V5 Gate C @50k:  MAE 0.01848, hit5 0.790, retrieval 1.000
```

V5 is the correct direction, but strict pass still fails:

```text
aggregate MAE < 0.020       pass
retrieval top1 = 1.000      pass
aggregate hit5 > 0.85       fail, measured 0.790
per-skin hit5 > 0.85        fail, 5 / 16 skins pass
```

The current best diagnosis is:

```text
V5 knows which skin it is seeing.
V5 can reconstruct low-frequency and moderate-detail skins.
V5 loses or smooths high-frequency local detail.
The failure is most visible on photographic/saturated skins.
```

## Architecture

Current V5 contract:

```text
input rendered PNG
-> shared CNN encoder
-> spatial encoder tokens + global style vector
-> per-file cross-attention heads
-> exact exported BMP tensors
-> Cranamp-supported-pixel loss/eval
```

Important implementation facts:

```text
input canvas                 1728 x 960
encoder final feature map    108 x 60
downsample factor            16x in both axes
encoder token count          6480
encoder token dim            128
style dim                    192
attention heads              4
cross-attention layers        1
query grid divisor           8
head channels                96
```

The final encoder feature map is not raw image evidence. It is a learned,
downsampled representation after four stride-2 stages. Each final token
covers a coarse area of the input. The per-file heads cross-attend into
those tokens, then decode from a low-resolution query grid using nearest
upsampling and convolution.

This is much better than V4's global-style-only heads, but it is still a
generator with a compressed middle. It is not a U-Net, not a pixel-aligned
copy path, and not an explicit inverse render/sampler.

## Why The User's Attention Concern Is Valid

The V5 debug attention maps are summaries:

```text
MultiheadAttention returns weights over encoder tokens.
The implementation averages attention over heads.
The debug path then averages over every query token in the file head.
The final heatmap is one coarse per-file map over the 108 x 60 encoder grid.
```

That means an attention overlay answers only this weak question:

```text
"Which broad input regions did this file head use on average?"
```

It does not answer:

```text
"Did the VOLUME slider output pixel use the VOLUME slider input pixel?"
"Did this exact output patch copy/preserve the right source patch?"
"Did every query in the file attend to the right local region?"
```

The user-provided VOLUME overlay for tvxq shows why this matters. The overlay
partly lands on the portrait/background, not cleanly on the volume slider
strip. So the previous wording "attention lands on the right regions" was too
strong. The corrected statement is:

```text
V5 is not completely lost, but averaged attention maps do not prove reliable
local source preservation. Some file heads, including tvxq VOLUME, can attend
to broad appearance context rather than the exact UI source region.
```

`REPORT_V5_GATEC_TVXQ.md` has been updated to reflect this correction.

## Gate A: BlueCurve One-Skin Overfit

Report:

```text
REPORT_V5_GATEA_BLUECURVE.md
```

Selected checkpoint:

```text
runs/slotnet_v5_bluecurve_gateA/snapshot_step020000.safetensors
```

Results:

```text
exported_pixels_mae        0.002118
exported_pixels_hit_5_255  0.992365
exported_pixels_sobel_mae  0.002956
```

Every trainable file had MAE below 0.01 and hit5 above 0.94. This is a real
one-skin overfit pass. Compared to V3.5 BlueCurve, V5 was about 4x lower MAE
and fixed the former large-panel weaknesses.

Interpretation:

```text
The V5 model/loss/export path is capable of representing and memorizing one
skin when the identity burden is trivial.
```

Gate A does not prove multi-skin capacity or generalization.

## Gate B: Three-Skin Overfit

Report:

```text
REPORT_V5_GATEB_3SKIN.md
```

Dataset:

```text
BlueCurve + DarkSide + Zelda
96 rows total, 32 variants per skin
```

Selected checkpoint:

```text
runs/slotnet_v5_3skin_gateB/snapshot_step020000.safetensors
```

Results:

```text
retrieval top1                 1.000 (96 / 96)
exported_pixels_mae            0.00554
exported_pixels_hit_5_255      0.971
exported_pixels_sobel_mae      0.00956
.wsz exports clean             3 / 3
```

Gate B was a strong pass. The recurring caveats were:

```text
Zelda BALANCE  MAE 0.054, hit5 0.889
Zelda EQMAIN   hit5 0.780
```

Interpretation:

```text
V5 can separate and reconstruct a few known identities well. The saturated
Zelda files were early warnings for the hard-skin pattern seen in Gate C.
```

## Gate C: Sixteen-Skin Memorization

Report:

```text
REPORT_V5_GATEC_16SKIN.md
```

Dataset:

```text
data_v4_16skin/train.csv
16 skins x 32 variants = 512 rows
```

Selected checkpoint:

```text
runs/slotnet_v5_16skin_gateC/snapshot_step050000.safetensors
```

Results:

```text
retrieval top1                    1.000 (512 / 512)
aggregate exported_pixels_mae     0.01848
aggregate hit_5_255               0.790
per-skin hit5 > 0.85              5 / 16
.wsz exports clean                6 / 6 selected exports
```

Comparison to V4:

```text
metric             V4 Gate 3 @50k    V5 Gate C @50k
retrieval top1     1.000             1.000
MAE                0.04065           0.01848
hit5               0.668             0.790
sobel              0.05852           0.02396
```

Interpretation:

```text
V5 solves identity separation and materially improves reconstruction.
V5 does not yet solve strict per-pixel reconstruction at 16 skins.
```

## Gate C Per-Skin Pattern

Strong skins:

```text
goodgawd                 hit5 0.978
simblyblayit             hit5 0.974
minimalistic_black       hit5 0.973
engraved4_platinum       hit5 0.948
darkside                 hit5 0.884
```

Near misses:

```text
infected_fx              hit5 0.837
cyborg                   hit5 0.829
ruki2                    hit5 0.825
rancid_amp               hit5 0.818
aguileramp               hit5 0.815
the_four_horsemen        hit5 0.812
```

Hard skins:

```text
a_halo_so_bright         hit5 0.714
dragonzv30amp            hit5 0.673
zelda_amp_gold           hit5 0.633
blair_razor_project      hit5 0.632
tvxq                     hit5 0.297
```

The solved skins are mostly flat-region or moderate-detail skins. The hard
skins are saturated, photographic, or high-frequency. This is a content
difficulty pattern, not identity collapse.

## Gate C Per-File Pattern

Files that remain hard at 16 skins:

```text
file        MAE        hit5
eqmain      0.03439    0.681
main        0.03099    0.692
volume      0.02648    0.702
balance     0.02256    0.783
pledit      0.01993    0.758
posbar      0.01780    0.781
cbuttons    0.01548    0.780
```

Files that are mostly solved:

```text
playpaus    0.00222    0.985
monoster    0.00733    0.910
titlebar    0.01189    0.844
```

The large or visually dense files are the bottleneck. Small sprites and
simple files are not the main problem.

## tvxq Follow-Up

Report:

```text
REPORT_V5_GATEC_TVXQ.md
```

Debug script:

```text
scripts/15_debug_tvxq.py
```

Corrected diagnosis:

```text
data corruption:        not found
identity collapse:      no, retrieval is perfect
attention/local path:   mixed, averaged maps are not proof of exact local use
capacity/detail:        yes, high-frequency photographic content is softened
```

Supported-pixel comparison between tvxq and minimalistic_black:

```text
file        tvxq     minimalistic_black   ratio
POSBAR      0.0701   0.0018               38.5x
MAIN        0.0749   0.0023               32.1x
VOLUME      0.0863   0.0033               26.2x
TITLEBAR    0.0508   0.0034               15.1x
EQMAIN      0.0724   0.0073                9.9x
BALANCE     0.0690   0.0072                9.6x
PLEDIT      0.0677   0.0080                8.5x
SHUFREP     0.0603   0.0078                7.7x
CBUTTONS    0.0727   0.0096                7.6x
MONOSTER    0.0286   0.0043                6.6x
PLAYPAUS    0.0030   0.0034                0.9x
```

The model roughly gets composition and color, but softens edges, text,
hair, and dense photographic texture. Under hit5, that softness fails most
pixels even when the result is visually recognizable.

## Main Architectural Concern

The user's concern is correct: V5 still compresses the image through a
middle representation.

The most important compression points:

```text
1. The shared encoder reduces 1728 x 960 input to 108 x 60 tokens.
2. Each token is 128D after projection.
3. Per-file heads receive one low-resolution query grid, not raw image pixels.
4. Decoders reconstruct from attended token features, not from high-res skips.
5. Attention debugging is averaged and not per-output-pixel.
```

This architecture can learn "what skin is this?" and "what should the file
roughly look like?" It is weaker at "copy this exact high-frequency local
detail into this exact output bitmap."

This explains the result pattern:

```text
identity retrieval: perfect
low-frequency skins: strong
photographic/high-frequency skins: soft
large files: weak
small simple files: solved
```

## What Not To Do Next

Do not treat V5 Gate C as a strict pass.

Do not rent a large GPU just to continue the same V5 training. The 30k to
50k slope is already shallow, and hit5 is mostly failing because of detail
loss, not identity confusion.

Do not rely on file weights as the main fix. File weights can move
priorities, but they do not add missing local evidence.

Do not use averaged attention overlays as proof that source preservation is
working.

Do not return to padded atlas training or priors. The exported-BMP contract
is one of the strongest improvements in the project.

## Recommended Next Experiments

### 1. Better Attention Debugging

Before changing the model, improve the debugging:

```text
- save per-query attention maps for selected output positions
- save maps for specific files and query cells, not only file-average maps
- for VOLUME/BALANCE/POSBAR, inspect query cells along the slider strip
- compare attention to expected UI regions qualitatively
```

This tells us whether the local path is misrouting or whether the decoder
receives enough evidence but cannot render it sharply.

### 2. Multi-Scale Local Evidence

The biggest architectural improvement is to expose higher-resolution encoder
features to the heads:

```text
current:
  file query grid -> cross-attend only to enc5 tokens (108 x 60)

candidate:
  file query grid -> cross-attend to enc5 for semantics
  file query grid -> cross-attend or sample enc3/enc4 for detail
  decoder receives fused multi-scale features
```

This is closer to a U-Net idea without returning to full-atlas generation.
It directly attacks the compression issue.

### 3. Denser Query Grid

Current query grid divisor:

```text
ceil(H / 8) x ceil(W / 8)
```

Probe:

```text
QUERY_GRID_DIVISOR 8 -> 4
```

This gives the decoder more file-local tokens and may reduce smoothing. It
also raises memory/compute. This is a useful controlled probe, especially
on Gate C or a 4-5 hard-skin subset.

### 4. Head Capacity Bump

Probe:

```text
head_channels 96 -> 128 or 160
add one or two residual conv blocks in the per-file decoder
```

This tests whether the decoder is capacity-bound after attention. It is
cheaper than redesigning the encoder, but it will not recover detail that
never reaches the head.

### 5. Loss Improvements After Architecture

Only after confirming the evidence path is adequate:

```text
- stronger edge/frequency loss
- perceptual feature matching
- small per-file discriminator
```

Loss can punish blur, but if the architecture discarded local detail at the
encoder bottleneck, a sharper loss alone can produce artifacts rather than
correct source preservation.

## Proposed V5.1 Gate Plan

Use the same gates, but add a hard-skin mini-gate before full Gate C:

```text
Gate A: BlueCurve one-skin
  must stay near V5 result: MAE <= 0.003, hit5 >= 0.98

Gate B: three-skin
  must stay near V5 result: retrieval 1.0, MAE <= 0.007, hit5 >= 0.95

Gate B-hard: 4 or 5 hard skins
  tvxq, Zelda, a_halo, dragonzv30amp, blair_razor
  target: clear hit5 lift over V5, especially MAIN/EQMAIN/VOLUME/BALANCE

Gate C: original 16-skin set
  target: retain retrieval 1.0, MAE < 0.02, hit5 materially above 0.790
```

Do not optimize only on tvxq. It is useful as an outlier probe, but a fix
that only memorizes tvxq is not useful.

## Bottom Line

V5 is a strong baseline and should replace V4 as the current architecture.
It fixed the most damaging previous mistakes and substantially improved the
16-skin result.

But V5 is still a compressed generator, not a maximum-preservation inverse
renderer. The user's attention concern is valid: averaged attention overlays
do not prove local source preservation, and the tvxq VOLUME overlay shows a
real reason to be cautious.

The next useful work is not more V5 training at the same settings. It is a
V5.1 architecture/debug pass focused on local high-resolution evidence:

```text
multi-scale encoder features
per-query attention debugging
denser query grid
larger/residual file decoders
```

Only after that should we revisit larger training runs or GPU rental.
