# Plan: SlotNet V6 Source-Preserving Inverse Renderer

Date: 2026-05-22

## Why V6

V5 should remain the current baseline, but not the final architecture.

V5 proved:

```text
identity retrieval works
exported-BMP training contract works
per-file cross-attention beats global-style heads
```

V5 did not prove:

```text
maximum source preservation
photographic/high-frequency reconstruction
strict per-skin hit5 at 16 skins
```

The failure mode is now clear:

```text
V5 recognizes the skin but smooths visible high-frequency detail.
```

V6 changes the mechanism from generation to learned inverse sampling:

```text
visible source pixels -> copied/sampled into BMP coordinates
hidden pixels         -> completed/generated
```

## V6 Output Per Exported BMP

For every trainable exported BMP file:

```text
uv_grid          [B, 2, H, W]
copy_conf_logits [B, 1, H, W]
copy_residual    [B, 3, H, W]
fallback_logits  [B, 3, H, W]
final_rgb        [B, 3, H, W]
```

Composition:

```python
copy_rgb = grid_sample(input_view, uv_grid.permute(0, 2, 3, 1), align_corners=False)
copy_refined = clamp(copy_rgb + residual_scale * tanh(copy_residual), 0, 1)
fallback_rgb = sigmoid(fallback_logits)
alpha = sigmoid(copy_conf_logits)
final_rgb = alpha * copy_refined + (1 - alpha) * fallback_rgb
```

The `grid_sample` path is the important difference from V5.

## Provenance First

V6 depends on correct synthetic labels. The first implementation milestone is
final-frame provenance in:

```text
cranamp_cli/cranamp/tools/cranamp_cli.py
```

The provenance buffer must describe what clean exported BMP pixel actually
survives into the final rendered input frame after overdraw, fills, text, and
window compositing.

Encoding:

```python
id = 1 + (file_id << 22) + (src_y << 11) + src_x
```

Rules:

```text
transparent/magenta pixels: no provenance write
sprite overdraw: later provenance overwrites earlier provenance
procedural fill/text/erase: provenance becomes 0
window stretching: provenance stretches with nearest-neighbor
transparent window pixels: do not clear existing final provenance
```

## Labels

Per synthetic sample:

```text
labels/{sample_id}.npz
```

Per file:

```text
visible_mask uint8 [H, W]
uv_target    float16 or uint16 [2, H, W]
```

Clean RGB targets stay the existing clean exported BMP pixels. Do not train
against post-composited screen RGB.

Use `align_corners=False` coordinates for `uv_target`:

```python
u = 2.0 * ((screen_x + 0.5) / INPUT_W) - 1.0
v = 2.0 * ((screen_y + 0.5) / INPUT_H) - 1.0
```

## Loss

Initial V6 loss:

```text
L = 1.00 * final_l1
  + 1.50 * final_sobel
  + 1.00 * copy_rgb
  + 0.25 * copy_conf_bce
  + 0.10 * uv_smooth_l1
  + 0.01 * uv_tv
```

Report:

```text
final MAE / hit5 / sobel
visible copy MAE
hidden MAE
copy_conf AUC
UV median pixel error
per-file metrics
```

## Gate Order

```text
Stage 0: provenance unit tests
Stage 1: oracle-copy sanity
Stage 2: copy head only, one skin
Stage 3: oracle observed completer
Stage 4: end-to-end V6 one-skin
Stage 5: hard-skin mini-gate
Stage 6: original 16-skin Gate C
```

Do not skip Stage 0 or Stage 1. If labels are wrong, V6 training will be
misleading.

## Hard-Skin Mini-Gate

Use the skins that exposed V5's weakness:

```text
tvxq_winamp_skins_by_roseweedy
zelda_amp_gold
a_halo_so_bright_it_bleeds
dragonzv30amp
blair_razor_project
```

Primary files to inspect:

```text
MAIN
EQMAIN
PLEDIT
VOLUME
BALANCE
CBUTTONS
```

V6 should materially improve copy-visible detail on these before full Gate C.

## Non-Goals

```text
no bigger same-V5 training
no prior atlas
no distortion metadata input
no user-provided mask input
no padded atlas pass/fail
no full-corpus training before staged gates
```

## Open Design Questions

1. Whether V6 needs multi-scale encoder features from enc3/enc4 immediately,
   or whether UV/copy from enc5 semantics is enough for a first probe.
2. Whether dense `.npz` labels are acceptable for all local experiments, or
   sparse labels should be implemented immediately.
3. Whether fallback completion is a separate module first, or integrated in
   the first end-to-end V6 model after copy-head sanity passes.

The recommendation is conservative:

```text
provenance -> labels -> oracle sanity -> copy head -> completer -> end-to-end
```
