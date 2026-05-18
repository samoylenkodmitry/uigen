# SlotNet V3.5 BlueCurve Overfit Report

Date: 2026-05-18

## Conclusion

V3.5 is the first successful overfit of the simplified Cranamp skin pipeline.
The important change was training and evaluating only the actual exported BMP
pixels instead of the padded 1024x1024 atlas.

The previous V3.4 "pass" was misleading because full-atlas MAE was diluted by
unused atlas space. V3.5 passes on the corrected exported-pixel criteria.

## Contract

```text
rendered input PNG -> predicted exported BMP tensors -> expected exported BMP pixels
```

No prior atlas, observed auxiliary head, dynamic mask, replay params, distortion
JSON, or full-atlas training objective participates in V3.5 training.

## Final Run

Run directory:

```text
runs/slotnet_v35_bluecurve_overfit/
```

Training command:

```bash
.venv/bin/python train_slotnet.py \
  --train data_v35_bluecurve_overfit/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --edge-weight 1.5 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --out runs/slotnet_v35_bluecurve_overfit \
  --device cuda
```

Dataset:

```text
32 distorted rendered input views
1 clean BlueCurve target atlas
```

## Corrected Metrics

Acceptance criteria:

```text
exported_pixels_mae < 0.02
exported_pixels_hit_5_255 > 0.85
```

`last.safetensors` all-row eval:

```text
exported_pixels_mae        0.00849
exported_pixels_hit_5_255  0.90394
exported_pixels_sobel_mae  0.00595
```

`best.safetensors` all-row eval:

```text
exported_pixels_mae        0.00846
exported_pixels_hit_5_255  0.90721
exported_pixels_sobel_mae  0.00612
```

The first full-eval pass occurred by step 10000:

```text
exported_pixels_mae        0.01147
exported_pixels_hit_5_255  0.86468
exported_pixels_sobel_mae  0.00845
```

## Remaining Weak Files

Final `best.safetensors` per-file metrics:

```text
PLEDIT    MAE 0.02120, hit 0.67920
EQMAIN    MAE 0.02451, hit 0.76079
MAIN      MAE 0.01472, hit 0.80974
TITLEBAR  MAE 0.01136, hit 0.86128
VOLUME    MAE 0.00714, hit 0.92484
```

Large/high-detail panels are still the hardest files. Small sprite sheets
overfit nearly exactly.

## Exports

Verified with `unzip -t`:

```text
runs/slotnet_v35_bluecurve_overfit/export_best/skin.wsz
runs/slotnet_v35_bluecurve_overfit/export_last/skin.wsz
```

Export folders contain only supported generated BMPs, runtime fallback
`TEXT.bmp`/`NUMBERS.bmp`, skin text files, `atlas.png`, and `skin.wsz`.
No `VIDEO.bmp`, `GEN.bmp`, replay params, target-view preview, or distortion
side-channel files are produced.

## Next Step

Use V3.5 as the baseline for broader experiments. Do not return to full-atlas
MAE or padded-atlas training as a pass/fail criterion.

Before multi-skin training, the next useful checks are:

```text
1. Repeat one-skin overfit on 2-3 very different skins.
2. Track per-file metrics, especially PLEDIT and EQMAIN.
3. Only then run a small multi-skin split.
```

