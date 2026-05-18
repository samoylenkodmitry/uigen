# Handoff For Claude: BlueCurve SlotNet V3.5 Overfit

Goal: prove the simplified pipeline can overfit one supported skin. Every
distorted input view should map back to the same clean expected BlueCurve skin
assets.

Do not start multi-skin training yet. This is only a one-skin overfit proof for
the corrected V3.5 objective.

## Current Code Contract

- Model: `models/slotnet_v35.py`
- Trainer: `train_slotnet.py`
- Loss: exported BMP pixel L1 plus Sobel edge loss in `models/losses.py`
- Direct evaluator: `scripts/09_eval_slotnet_overfit.py`
- Static trainable export spec: `atlas_ai/export_spec.py`
- Training atlas profile: `configs/atlas_train_v1.json`
- Export profile: `configs/export_profile_classic.json`
- Source skin: `skins_raw/BlueCurve_Winamp.wsz`

Do not use pre-V3.5 datasets, old checkpoints, prior atlases, observed heads,
visible masks, rect/state side channels, replay params, debug contact sheets,
or preview renders as training inputs.

## Invariants

- Input to SlotNet is only the rendered RGB PNG tensor.
- Output from SlotNet V3.5 is only predicted tensors for trainable exported
  BMPs.
- Target is only the expected atlas PNG, cropped internally to exact exported
  BMP dimensions.
- Distortions exist only in the rendered input PNG pixels.
- Dataset CSV rows should contain only:
  `skin_id, variant_id, view_png, atlas_png, meta_json`.
- The BlueCurve overfit dataset should have many distorted views, but all rows
  must point to the same expected atlas PNG.
- Exported skin folder should stay focused on skin files plus `atlas.png` and
  `skin.wsz`; do not write load-smoke preview PNGs there.

## Pre-Flight

Run the unit suite before training:

```bash
.venv/bin/python -m pytest -q
```

Expected:

```text
25 passed
```

Confirm the active trainer is V3.5-only:

```bash
.venv/bin/python train_slotnet.py --help | grep -E "edge-weight|runs/slotnet_v35"
```

Do not add a `--model-version 34` option back to the trainer. V3.4 checkpoint
support is only for inspecting old runs through `infer_skin.py` and
`scripts/09_eval_slotnet_overfit.py`.

## Build Fresh BlueCurve Dataset

Use a fresh dataset root; do not train from old V3.4 output folders.

```bash
mkdir -p /tmp/uigen_v35_bluecurve_raw
ln -sf /home/s/develop/projects/uigen/skins_raw/BlueCurve_Winamp.wsz /tmp/uigen_v35_bluecurve_raw/skin.wsz

.venv/bin/python scripts/01_pack_skins.py \
  --skins-raw /tmp/uigen_v35_bluecurve_raw \
  --atlas-profile configs/atlas_train_v1.json \
  --out data_v35_bluecurve_overfit

.venv/bin/python scripts/02_render_dataset.py \
  --valid-skins data_v35_bluecurve_overfit/valid_skins.csv \
  --out data_v35_bluecurve_overfit \
  --variants 32 \
  --workers 1 \
  --cranamp-cli ./cranamp_cli/cranamp-cli

.venv/bin/python scripts/03_make_splits.py \
  --data data_v35_bluecurve_overfit \
  --valid-skins data_v35_bluecurve_overfit/valid_skins.csv

.venv/bin/python scripts/04_check_dataset.py \
  --data data_v35_bluecurve_overfit
```

Expected dataset shape:

```bash
find data_v35_bluecurve_overfit -maxdepth 2 -type f | sort
```

There should be `views/`, `atlases/`, CSVs, and metadata only. No `debug/`,
`params/`, `rects/`, `states/`, `visible_masks/`, or replay JSON files.

The rendered input PNGs are allowed to look distorted. That is the input. The
expected target is the clean packed atlas referenced by `atlas_png`.

## Train One-Skin Overfit

Start with base 24. If CUDA memory fails, drop to base 16, but do not change the
data contract.

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

Recommended launch wrapper:

```bash
nohup .venv/bin/python train_slotnet.py \
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
  --device cuda \
  > /tmp/uigen_v35_bluecurve_train.log 2>&1 &
echo "pid=$!"
```

Do not use old `runs/slotnet_v34_*` checkpoints as warm starts.

## Monitor During Training

Every 30 minutes, inspect the training log and latest metric row:

```bash
tail -40 /tmp/uigen_v35_bluecurve_train.log
tail -1 runs/slotnet_v35_bluecurve_overfit/metrics.jsonl
```

The metric keys that matter during training are:

- `exported_l1`
- `exported_sobel`
- `exported_hit5`
- per-file `mae_*`
- per-file `sobel_*`

`total` is the weighted optimization loss. Use it for trend, not acceptance.

If any individual exported file is flat while the average improves, stop and
debug before continuing.

## Measure Overfit

Use exported BMP metrics as pass/fail. Full 1024x1024 atlas MAE is debug-only
because padded atlas space dilutes the error.

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_bluecurve_overfit/train.csv \
  --slotnet runs/slotnet_v35_bluecurve_overfit/last.safetensors \
  --device cpu
```

Acceptance target for this one-skin overfit:

- `exported_pixels_mae < 0.02`: acceptable first pass.
- `exported_pixels_mae < 0.01`: strong evidence the model can memorize the
  expected exported skin pixels.
- `exported_pixels_hit_5_255 > 0.85`: small sprites are not just roughly
  colored.
- Review `per_exported_file`; no supported file should silently fail while the
  average looks good.

If exported-pixel metrics plateau above target, stop and debug the simple path
before adding priors, masks, or auxiliary heads.

Do not report the run as passed from `full_atlas_mae`.

## Export A Checkpoint

Export one distorted input view through the trained checkpoint. This creates the
actual `.wsz` for manual Cranamp loading:

```bash
VIEW=$(tail -n +2 data_v35_bluecurve_overfit/train.csv | cut -d, -f3 | head -1)

.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v35_bluecurve_overfit/last.safetensors \
  --out runs/slotnet_v35_bluecurve_overfit/export_last \
  --device cpu

unzip -t runs/slotnet_v35_bluecurve_overfit/export_last/skin.wsz
```

Expected export folder: generated BMPs, `atlas.png`, and `skin.wsz`. It should
include `PLEDIT.bmp` and should not include `VIDEO.bmp`.

Load checks may render to `/tmp`, but do not leave random-render preview PNGs in
the export folder because they look intentionally distorted and confuse review.

For interim snapshot review, replace `last.safetensors` with a snapshot:

```bash
.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v35_bluecurve_overfit/snapshot_step010000.safetensors \
  --out runs/slotnet_v35_bluecurve_overfit/export_step010000 \
  --device cpu

unzip -t runs/slotnet_v35_bluecurve_overfit/export_step010000/skin.wsz
```

Do not create `predicted_render_replay_input_params`, `target_view.png`,
contact sheets, or random-render previews in the export directory.

## Stop Conditions

Stop training and report instead of continuing if:

- `exported_pixels_mae` remains above `0.02` after a full one-skin run.
- Any trainable file has much worse `per_exported_file.*.mae` than the rest.
- The exported `.wsz` cannot be opened or `unzip -t` fails.
- The export folder contains unsupported generated BMPs such as `VIDEO.bmp` or
  `GEN.bmp`.
- Someone suggests adding priors, masks, replay params, observed heads, or
  distortion JSON back into the pipeline.
