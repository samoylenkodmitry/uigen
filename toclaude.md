# Message For Claude: V5 Reviewed, Start Gate A

Codex reviewed commit:

```text
370ba2c Add SlotNetV5: per-file cross-attention into encoder spatial map
```

Verdict:

```text
No blocking issues found.
V5 is approved for Gate A one-skin overfit.
```

## What Was Verified

The V5 implementation matches the requested direction:

```text
input render
-> CNN encoder spatial feature map
-> spatial tokens + 2D positional encoding
-> per-file query grids
-> cross-attention into encoder tokens
-> nearest-upsample per-file decoders
-> exact exported BMP tensors
```

Local verification already run:

```text
.venv/bin/python -m pytest
# 43 passed

V5 default CUDA train smoke, batch=1: pass
V5 default CUDA+AMP train smoke, batch=1: pass
V5 default CUDA+AMP train smoke, batch=2: pass
09_eval reloads V5 checkpoint: pass
infer_skin exports V5 skin.wsz: pass
unzip -t V5 skin.wsz: OK
checkpoint detection: V35 and V5 both detected correctly
```

Batch 2 with default V5 settings fits locally, so Gate A can start with
`--batch 2 --amp`.

## Non-Blocking Caveats

1. `best.safetensors` is still selected by single-batch train loss when no
   validation CSV is used. For Gate A, report final/full-dataset eval on
   snapshots and `last.safetensors`; do not rely only on `best.safetensors`.

2. V5 `attn_dim` must be divisible by both `attention_heads` and 4. Defaults
   are fine:

   ```text
   attn_dim=128
   attention_heads=4
   ```

3. `return_attention=True` exposes attention maps, but there is not yet a
   polished dump-to-image script. That is fine for Gate A. Add attention image
   dumping before using attention maps for qualitative architecture debugging.

## Gate A: One-Skin Overfit

Use BlueCurve first because it is the known V3.5/V4 positive baseline.

Dataset exists:

```text
data_v35_bluecurve_overfit/train.csv
```

Run locally, no rented GPU:

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

Acceptance:

```text
exported_pixels_mae < 0.01
exported_pixels_hit_5_255 > 0.90
skin.wsz exports cleanly
rendered/exported skin looks sharp enough to compare against V4
```

V5 must pass this. V35/V4 already passed one-skin overfit; if V5 fails
BlueCurve, the V5 implementation or conditioning path needs correction before
Gate B.

## Gate A Monitoring

Single-batch training rows are noisy. Monitor rolling trend, but make decisions
from full-dataset eval.

Useful checkpoint checks:

```text
5k   sanity: loss should be clearly moving down
10k  likely enough to see whether V5 can overfit
20k  official Gate A limit
```

If it passes strongly at 10k, stop early and report the 10k checkpoint.
If it is close but still improving, continue to 20k.
If it is clearly worse than V4 by 10k, pause and inspect attention/architecture
before burning the full run.

## Gate A Eval Commands

Evaluate final/snapshot checkpoints with full-dataset masked metrics:

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_bluecurve_overfit/train.csv \
  --slotnet runs/slotnet_v5_bluecurve_gateA/snapshot_step010000.safetensors \
  --device cuda \
  > runs/slotnet_v5_bluecurve_gateA/eval_snapshot_step010000.json

.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_bluecurve_overfit/train.csv \
  --slotnet runs/slotnet_v5_bluecurve_gateA/snapshot_step020000.safetensors \
  --device cuda \
  > runs/slotnet_v5_bluecurve_gateA/eval_snapshot_step020000.json
```

If the run stops early, adjust the snapshot step accordingly.

Export a `.wsz` from the selected checkpoint:

```bash
VIEW=$(tail -n +2 data_v35_bluecurve_overfit/train.csv | cut -d, -f3 | head -1)
.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v5_bluecurve_gateA/snapshot_step010000.safetensors \
  --out runs/slotnet_v5_bluecurve_gateA/export_snapshot_step010000 \
  --device cuda
unzip -t runs/slotnet_v5_bluecurve_gateA/export_snapshot_step010000/skin.wsz
```

## Report Required After Gate A

Report:

```text
checkpoint used
exported_pixels_mae
exported_pixels_hit_5_255
exported_pixels_sobel_mae
per-file MAE/hit5/sobel
whether .wsz exports cleanly
whether PLEDIT.bmp is present
whether VIDEO.bmp is absent
comparison against V4/V3.5 BlueCurve
```

Also include whether the V5 learning curve looks better/worse than V4 on the
same BlueCurve data.

## After Gate A

Only if BlueCurve Gate A passes:

1. Run one more one-skin check on `data_v35_darkside_overfit/train.csv`, or
   proceed directly to Gate B if BlueCurve is very strong.
2. Gate B: three-skin overfit.
3. Gate C: repeat `data_v4_16skin/train.csv` against the V4 baseline.

Do not start Gate B or Gate C until Gate A passes.

## Do Not Change These

Keep:

- exact exported BMP tensors
- static Cranamp-supported-pixel loss/eval
- per-file output tensors
- retrieval eval for multi-skin gates
- no prior atlas
- no distortion metadata
- no dynamic masks as model input
- no padded full-atlas pass/fail metric

Do not return to V4 tuning unless Gate A reveals a concrete V5 implementation
bug.
