# Gate C Follow-Up — tvxq Failure Diagnosis

Date: 2026-05-22

## Question

Gate C had one clear outlier: `tvxq_winamp_skins_by_roseweedy_c379f7bd` with
aggregate hit5 0.297 and 10 of 11 files below 0.85 hit5. Is this a data
issue (bad render / bad target / wrong CSV row), an attention failure (model
not localizing the right input region), or a capacity/detail failure (decoder
too soft for the content)?

## Method

`scripts/15_debug_tvxq.py` runs V5 `snapshot_step050000` on the first row of
the bad skin (tvxq) and one control row of a clean skin
(`minimalistic_black_145917e6`, Gate C aggregate hit5 0.973). For each:

```text
- save letterboxed input view (what the model sees)
- save raw input view (what was rendered)
- per trainable file: target BMP crop, predicted BMP crop, abs-diff heatmap
- attention heatmap overlay on input view for MAIN, EQMAIN, PLEDIT, VOLUME, BALANCE
- per-file supported-pixel MAE matching Gate C training/eval, plus full-rectangle
  MAE for visual-debug context
```

Run:

```bash
.venv/bin/python scripts/15_debug_tvxq.py \
  --samples data_v4_16skin/train.csv \
  --slotnet runs/slotnet_v5_16skin_gateC/snapshot_step050000.safetensors \
  --out runs/slotnet_v5_16skin_gateC/debug_tvxq \
  --device cuda
```

Artifacts in `runs/slotnet_v5_16skin_gateC/debug_tvxq/{bad,good}/`.

## Findings

### 1. Data: not the issue.

The tvxq input view and target atlas are well-formed:

```text
view file:    data_v4_16skin/views/tvxq_winamp_skins_by_roseweedy_c379f7bd_0000.png
atlas file:   data_v4_16skin/atlases/tvxq_winamp_skins_by_roseweedy_c379f7bd.png
```

The atlas BMP slots crop correctly via `TRAINABLE_EXPORT_SPECS` (no
shape mismatches, no zero crops). The render shows a coherent skin layout.

What tvxq is: a fully **photographic** Winamp skin. The UI background of the
main window, equalizer, and playlist editor is a portrait photograph
(labelled "cutie da changmin" in the title bar). Slider tracks, button
labels, and song-list text are overlaid on top of the photo. The entire
trainable surface has high-frequency texture, with no flat-color regions.

By contrast, `minimalistic_black` is the inverse: solid blue panels with
small white sliders and dark text — almost all low-frequency, high-contrast
flat regions.

The render system and ground truth are working as intended; tvxq is
genuinely a hard skin, not a corrupted one.

### 2. Attention: not the issue.

Attention overlays for `MAIN` on tvxq concentrate on the face/hair region
of the input image — exactly where the photographic detail being
reconstructed lives. The attention maps for `EQMAIN`, `PLEDIT`, `VOLUME`,
`BALANCE` similarly hit relevant input regions (slider areas, button
strips, body of the panel). The model is not lost.

The minimalistic_black attention maps focus on slider edges and panel
boundaries — design-defining structure rather than flat-fill regions —
which makes sense for a flat-color skin.

V5's cross-attention path is working. It is not the bottleneck on tvxq.

### 3. Decoder capacity / detail: **this is the issue.**

The predicted MAIN on tvxq shows the correct overall composition:

```text
- correct face / hair silhouette
- correct title-bar label text position
- correct background color
- correct slider rhythm in the equalizer band
```

But high-frequency content is softened or smeared:

```text
- face edges are blurry
- track-listing text in PLEDIT collapses to illegible smears
- slider tracks in EQMAIN are recovered but the photographic background
  blends into broad color smears
- letter shapes are mean-color blobs, not crisp glyphs
```

This is the canonical "regression-to-mean blurry prediction" failure mode
of pixel L1-style training on high-frequency targets. Numerically, the
supported-pixel MAE from the Gate C evaluator shows tvxq is much harder
than the solved `minimalistic_black` control on the pixels Cranamp actually
renders:

```text
supported MAE      tvxq    minimalistic_black   ratio
POSBAR             0.0701  0.0018               38.5x
MAIN               0.0749  0.0023               32.1x
VOLUME             0.0863  0.0033               26.2x
TITLEBAR           0.0508  0.0034               15.1x
EQMAIN             0.0724  0.0073                9.9x
BALANCE            0.0690  0.0072                9.6x
PLEDIT             0.0677  0.0080                8.5x
SHUFREP            0.0603  0.0078                7.7x
CBUTTONS           0.0727  0.0096                7.6x
MONOSTER           0.0286  0.0043                6.6x
PLAYPAUS           0.0030  0.0034                0.9x
```

The earlier full-rectangle diagnostic produced the same broad direction,
but it was weaker evidence because unsupported regions dominate some BMPs
such as `TITLEBAR.bmp` and `PLEDIT.bmp`. The supported-pixel table above
is the canonical one.

Per-pixel hit5 (5/255 tolerance) is unforgiving of any softness. A
prediction that is visually-close-but-mean-blurred fails per-pixel for
every soft edge, while flat-region skins like minimalistic_black pass
trivially because there is no high-frequency content to soften in the
first place.

This also explains the rest of the Gate C hard-skin pattern:

```text
zelda_amp_gold        saturated gold textures   (high-freq)
blair_razor_project   saturated reds            (high-freq edges)
dragonzv30amp         busy color gradients      (high-freq)
a_halo_so_bright      bright photographic       (high-freq)
```

Every hard skin has high-frequency / photographic / saturated content.
Every solved skin (minimalistic_black, goodgawd, simblyblayit, engraved4,
darkside) is flat-region-dominated. The mechanism is the same on tvxq as
on the other hard skins; tvxq is the most extreme case (entirely
photographic).

## Diagnosis Summary

```text
data issue              NO   inputs/targets well-formed
attention failure       NO   attention reaches relevant input regions
identity / retrieval    NO   tvxq has unique global style; retrieval top1 = 1.000
capacity / detail bound YES  decoders produce mean-color soft prediction;
                             fails per-pixel hit5 even when visually close
```

This is the same fundamental limit that caps every saturated/photographic
skin in Gate C. tvxq is not architecturally special — it is the worst
case along a single axis (fraction of the trainable surface that contains
high-frequency content).

## Implications for Next Steps

Discarded:

```text
- "Investigate tvxq for data bug"           done; no bug
- "Investigate tvxq for attention bug"      done; attention works
- "Per-skin curriculum / up-sample tvxq"    won't fix the soft-prediction
                                            mechanism; would burn cycles
- "More training steps"                     30k -> 50k slope already shallow;
                                            mean-blur converges quickly and
                                            stays there
```

Worth considering:

```text
1. Capacity bump on the head decoders.
   - Larger head_channels (96 -> 128 or 160).
   - More residual blocks in the nearest-upsample path.
   - Memory cost is local to the heads; encoder shape unchanged.

2. Loss change to escape mean-blur.
   - Add a perceptual / feature-matching component (VGG-style or
     in-network features) so the loss penalises softness independent of
     per-pixel mean.
   - Or an adversarial discriminator on per-file outputs.
   - Both are real engineering work; both directly target the mechanism
     we observed.

3. Higher-resolution attention query grid.
   - Current per-file query grid is ceil(H/8) x ceil(W/8). On tvxq the
     correct content is at near-pixel granularity. A denser query grid
     gives the decoder more local evidence per output pixel.

4. Frequency-aware loss weighting.
   - Up-weight gradient / sobel terms on the per-file loss so high-
     frequency edges drive more learning. Currently edge_weight 1.5 is
     uniform across the batch; could be conditioned on per-skin /
     per-file local-frequency statistics.
```

Recommended order: 1 (cheapest, directly probes the capacity hypothesis),
then 3 (cheap, also directly addresses the soft-prediction mechanism), then
2 (expensive but the most principled fix).

Capacity bump and query-grid resolution both validate or kill the capacity
hypothesis at minimal cost. Perceptual / adversarial losses are large
follow-ups whose value depends on whether (1) and (3) have already moved
the needle.

## Artifacts

```text
scripts/15_debug_tvxq.py                              (committed)
runs/slotnet_v5_16skin_gateC/debug_tvxq/
  bad/00_input_letterboxed.png
  bad/00_input_raw.png
  bad/target_<11 files>.png
  bad/predicted_<11 files>.png
  bad/diff_<11 files>.png
  bad/attn_overlay_{MAIN,EQMAIN,PLEDIT,VOLUME,BALANCE}.png
  bad/attn_raw_*.npy
  bad/summary.json
  good/(same set for minimalistic_black)
  summary.json
```

Total artifact size ~3.5 MB; not committed (lives in gitignored `runs/`).
Regenerate any time with the script above.
