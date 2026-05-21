# SlotNet V5 Gate A — BlueCurve One-Skin Overfit

Date: 2026-05-21

## Conclusion

V5 passes Gate A on BlueCurve with substantial margin. Both the 10k snapshot
and the 20k snapshot exceed the acceptance criteria; the 20k snapshot is
strictly better on every aggregate metric and on every per-file metric, so it
is the selected checkpoint for downstream use.

V5 is approved to move to a second one-skin check (DarkSide) or directly to
Gate B.

## Acceptance vs. Result

Acceptance criteria:

```text
exported_pixels_mae        < 0.01
exported_pixels_hit_5_255  > 0.90
```

Selected checkpoint `snapshot_step020000.safetensors`:

```text
exported_pixels_mae        0.002118   PASS (4.7x under cap)
exported_pixels_hit_5_255  0.992365   PASS
exported_pixels_sobel_mae  0.002956
full_atlas_mae             0.036323   (informational; not a pass criterion)
```

10k checkpoint (also passes):

```text
exported_pixels_mae        0.003569
exported_pixels_hit_5_255  0.978547
exported_pixels_sobel_mae  0.004745
```

## Training Run

Run directory:

```text
runs/slotnet_v5_bluecurve_gateA/
```

Command (as specified in the V5 handoff):

```bash
.venv/bin/python train_slotnet.py \
  --model-version 50 \
  --train data_v35_bluecurve_overfit/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --head-channels 96 \
  --attn-dim 128 \
  --attention-heads 4 \
  --cross-attention-layers 1 \
  --file-embedding-dim 32 \
  --edge-weight 1.5 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v5_bluecurve_gateA \
  --device cuda
```

Hardware: single RTX 2070 (8 GB), AMP enabled, batch 2.

Wall time: ~2h18m for 20000 steps.

Final reported training loss: `0.003265` (single-batch noisy, do not infer
acceptance from this; metrics above are from full-dataset eval).

Dataset:

```text
data_v35_bluecurve_overfit/train.csv   32 distorted rendered input views
                                       1 clean BlueCurve target atlas
```

## Per-File Metrics — Selected Checkpoint (step 20000)

```text
file        MAE        hit_5_255   sobel_MAE
balance     0.002783   0.993958    0.002919
cbuttons    0.001568   0.999931    0.002360
eqmain      0.006644   0.947256    0.012054
main        0.004065   0.979317    0.004920
monoster    0.000521   1.000000    0.000741
playpaus    0.000487   1.000000    0.000889
pledit      0.002076   0.996923    0.002056
posbar      0.000791   1.000000    0.001170
shufrep     0.001203   1.000000    0.001315
titlebar    0.001127   1.000000    0.001606
volume      0.002035   0.998633    0.002490
```

Every file is well under the 0.01 MAE cap and well over the 0.90 hit rate
threshold. `eqmain` remains the relatively weakest file (consistent with V3.5),
but it is no longer close to failing — V5 brings it from 0.024 MAE / 0.761 hit
on V3.5 down to 0.0066 / 0.947.

## Comparison vs. V3.5 BlueCurve

V3.5 is the direct comparable (V4 ran Gate 2/Gate 3, not a 1-skin BlueCurve
overfit). Numbers from `REPORT_V35_BLUECURVE.md` (`best.safetensors`):

```text
metric                     V3.5 best     V5 step 20000   improvement
exported_pixels_mae        0.00846       0.00212         4.0x lower
exported_pixels_hit_5_255  0.90721       0.99237         +0.085 absolute
exported_pixels_sobel_mae  0.00612       0.00296         2.1x lower
```

Per-file, the V3.5 weakest files all improve sharply on V5:

```text
file        V3.5 MAE / hit5      V5 MAE / hit5         delta
PLEDIT      0.02120 / 0.6792     0.00208 / 0.9969      10.2x lower MAE
EQMAIN      0.02451 / 0.7608     0.00664 / 0.9473      3.7x lower MAE
MAIN        0.01472 / 0.8097     0.00407 / 0.9793      3.6x lower MAE
TITLEBAR    0.01136 / 0.8613     0.00113 / 1.0000      10.1x lower MAE
VOLUME      0.00714 / 0.9248     0.00204 / 0.9986      3.5x lower MAE
```

The per-file cross-attention into the encoder spatial map clearly helps the
large, high-detail panels — exactly where V3.5/V4 struggled most.

## Comparison vs. V4

V4 has no direct BlueCurve 1-skin overfit number, so a like-for-like comparison
is not possible. V4 Gate 2 (3-skin masked-loss overfit) succeeded; V4 Gate 3
(16-skin) showed identity-separation pass but reconstruction failure
(`REPORT_V4_GATE3.md`). V5 was designed in response to that V4 Gate 3
reconstruction gap.

Whether V5 closes the V4 Gate 3 reconstruction gap can only be answered by
Gate B (3-skin) and Gate C (16-skin) — Gate A passing is a necessary precondition,
not evidence that V5 fixes Gate 3.

## Learning Curve

10k → 20k aggregate progression:

```text
metric                     step 10000   step 20000   delta
exported_pixels_mae        0.003569     0.002118     -41%
exported_pixels_hit_5_255  0.978547     0.992365     +0.014
exported_pixels_sobel_mae  0.004745     0.002956     -38%
```

10k already passes Gate A with room. The handoff allows early-stop reporting
when the 10k checkpoint passes strongly. The 10k–20k window still produced a
material improvement on every metric — there was no sign of overfit
degradation, so the full 20k run is the better checkpoint to use as a baseline.

If wall time matters for repeat runs (DarkSide, Gate B), 10k is a defensible
early-stop budget on BlueCurve. Whether the same is true for other skins
should be checked empirically; do not assume.

## Export

`.wsz` exported from `snapshot_step020000`:

```text
runs/slotnet_v5_bluecurve_gateA/export_snapshot_step020000/skin.wsz
```

`unzip -t` result: `No errors detected in compressed data` for all 16 files.

Archive contents:

```text
BALANCE.bmp      62406 bytes
CBUTTONS.bmp     14742 bytes
EQMAIN.bmp      260874 bytes
MAIN.bmp         95274 bytes
MONOSTER.bmp      4086 bytes
NUMBERS.bmp       5204 bytes   (runtime fallback)
PLAYPAUS.bmp      1206 bytes
PLEDIT.bmp      156294 bytes   present
PLEDIT.TXT         117 bytes
POSBAR.bmp        9294 bytes
README.txt         162 bytes
SHUFREP.bmp      23514 bytes
TEXT.bmp         11216 bytes   (runtime fallback)
TITLEBAR.bmp     89838 bytes
VISCOLOR.TXT       599 bytes
VOLUME.bmp       88386 bytes
```

Required-file checks:

```text
PLEDIT.bmp present:   yes
VIDEO.bmp absent:     yes
```

No replay params, distortion side-channel, prior-atlas, full-padded-atlas
artifacts, or target-view previews are produced. The export matches the
restricted Cranamp surface.

## Eval Artifacts

```text
runs/slotnet_v5_bluecurve_gateA/eval_snapshot_step010000.json
runs/slotnet_v5_bluecurve_gateA/eval_snapshot_step020000.json
runs/slotnet_v5_bluecurve_gateA/export_snapshot_step020000/atlas.png
runs/slotnet_v5_bluecurve_gateA/export_snapshot_step020000/skin.wsz
```

## Next

Gate A passes. Per the V5 handoff, options:

```text
1. Run one-skin check on data_v35_darkside_overfit/train.csv.
2. Proceed directly to Gate B (three-skin overfit).
```

BlueCurve passed strongly enough (4.7x under the MAE cap, every per-file hit
above 0.94) that the DarkSide check is reasonable to skip in favor of Gate B,
unless the user wants a second sanity point. Recommendation: skip directly to
Gate B; if Gate B passes, the second 1-skin check would have been redundant,
and if Gate B fails, the DarkSide overfit would not have predicted the
multi-skin failure mode anyway.
