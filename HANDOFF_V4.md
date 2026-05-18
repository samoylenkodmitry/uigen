# Handoff For Claude: SlotNet V4 Gates

Goal: decide whether V3.5 can scale beyond one skin before renting GPU time.
This is not a new prior/mask/aux-head model. V4 means auditability, training
infrastructure, and multi-identity gates around the V3.5 exported-BMP contract.

## Current Baseline

V3.5 BlueCurve overfit passed the first real one-skin gate:

```text
best.safetensors exported_pixels_mae        0.00846
best.safetensors exported_pixels_hit_5_255  0.90721
```

That proves the corrected objective can memorize one skin. It does not prove
multi-skin identity separation or generalization.

Keep the contract:

```text
rendered input PNG -> predicted exported BMP tensors -> expected exported BMP pixels
```

Do not reintroduce full-atlas loss, prior atlases, observed heads, dynamic
masks, replay params, distortion JSON, target-view renders, contact sheets, or
unsupported BMPs.

## New Tooling In This Handoff

- `scripts/10_audit_export_dimensions.py`
  - Packs a source skin, exports supported BMPs, and reports source size vs
    export-profile size vs crop exactness.
  - Use this to verify the suspicious `MAIN.bmp 275x115` crop.
- `configs/file_weights_v35_large_panels.yaml`
  - Optional file-weight override that boosts `MAIN`, `EQMAIN`, `PLEDIT`, and
    `TITLEBAR`.
- `train_slotnet.py`
  - Adds `--style-dim`, `--head-channels`, `--file-weights-yaml`, `--resume`,
    `--val-csv`, `--eval-every`, `--num-workers`, `--pin-memory`,
    `--persistent-workers`, `--prefetch-factor`, and `--amp`.
  - Saves run metadata to `config.yaml`, including git hash, dataset summary,
    file weights, and best metric.
  - With `--val-csv --eval-every N`, saves `best.safetensors` by
    `val_exported_l1`.
- `scripts/11_eval_slotnet_retrieval.py`
  - Computes top-1 target retrieval over exported-pixel MAE for multi-skin
    memorization.
- `scripts/12_make_variant_split.py`
  - Splits a sample CSV by `variant_id`, usually variants `0000-0023` train and
    `0024-0031` validation.

## Pre-Flight

Run tests before any training:

```bash
.venv/bin/python -m pytest -q
```

Check trainer options:

```bash
.venv/bin/python train_slotnet.py --help | grep -E "file-weights-yaml|val-csv|num-workers|amp|resume"
```

## Gate 1: Export Dimension Audit

Run this on BlueCurve and every candidate one-skin test:

```bash
mkdir -p runs/audits

.venv/bin/python scripts/10_audit_export_dimensions.py \
  --skin skins_raw/BlueCurve_Winamp.wsz \
  --out-json runs/audits/bluecurve_export_audit.json
```

Read the table. For every supported BMP, `profile_ok` must be `True`,
`overlap_mae` should be `0.00000000`, and `pad_black` should be `0.00000000`
when the export profile is larger than the source BMP.

Important: BlueCurve currently reports source-size mismatches for `MAIN.bmp`,
`MONOSTER.bmp`, `PLEDIT.bmp`, `VOLUME.bmp`, and `BALANCE.bmp`, while the
overlap pixels match exactly. Do not silently call those a bug or a pass. Verify
whether Cranamp intentionally uses the configured exported dimensions,
especially `MAIN.bmp 275x115`. Document the conclusion in the next report. If
the profile is wrong, fix `configs/export_profile_classic.json`, regenerate
targets, and rerun BlueCurve before moving on.

## Gate 2: Diverse One-Skin Overfits

Before 16-skin training, run 3-5 one-skin overfits with different visual styles:

```text
1. BlueCurve / metallic smooth UI
2. dark photographic or textured skin
3. bright/light skin
4. low-color pixel-art skin
5. weird transparency or high-saturation skin
```

For each skin, build a fresh dataset exactly like V3.5:

```bash
mkdir -p /tmp/uigen_v4_one_raw
ln -sf /absolute/path/to/SKIN.wsz /tmp/uigen_v4_one_raw/skin.wsz

.venv/bin/python scripts/01_pack_skins.py \
  --skins-raw /tmp/uigen_v4_one_raw \
  --atlas-profile configs/atlas_train_v1.json \
  --out data_v35_SKINNAME_overfit

.venv/bin/python scripts/02_render_dataset.py \
  --valid-skins data_v35_SKINNAME_overfit/valid_skins.csv \
  --out data_v35_SKINNAME_overfit \
  --variants 32 \
  --workers 1 \
  --cranamp-cli ./cranamp_cli/cranamp-cli

.venv/bin/python scripts/03_make_splits.py \
  --data data_v35_SKINNAME_overfit \
  --valid-skins data_v35_SKINNAME_overfit/valid_skins.csv \
  --train 1.0 --val 0.0 --test 0.0
```

Train:

```bash
.venv/bin/python train_slotnet.py \
  --train data_v35_SKINNAME_overfit/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --edge-weight 1.5 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v35_SKINNAME_overfit \
  --device cuda
```

Evaluate:

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_SKINNAME_overfit/train.csv \
  --slotnet runs/slotnet_v35_SKINNAME_overfit/best.safetensors \
  --device cpu
```

Acceptance per skin:

```text
exported_pixels_mae < 0.02
exported_pixels_hit_5_255 > 0.85
skin.wsz loads in Cranamp
rendered output is visually sharp
```

If the same large files keep trailing (`PLEDIT`, `EQMAIN`, `MAIN`, `TITLEBAR`),
run one short comparison, not an endless tuning loop:

```bash
.venv/bin/python train_slotnet.py \
  --train data_v35_SKINNAME_overfit/train.csv \
  --steps 5000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --edge-weight 1.5 \
  --file-weights-yaml configs/file_weights_v35_large_panels.yaml \
  --resume runs/slotnet_v35_SKINNAME_overfit/best.safetensors \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v35_SKINNAME_overfit_largepanel_ft \
  --device cuda
```

Keep the fine-tune only if large-panel metrics improve without damaging small
sprites.

## Gate 3: 16-Skin Memorization

Only after diverse one-skin overfits pass, create a 16-skin dataset with 32
variants per skin. This is a memorization test, so train on all rows:

```bash
.venv/bin/python scripts/03_make_splits.py \
  --data data_v35_16skin \
  --valid-skins data_v35_16skin/valid_skins.csv \
  --train 1.0 --val 0.0 --test 0.0
```

Train up to 50k steps:

```bash
.venv/bin/python train_slotnet.py \
  --train data_v35_16skin/train.csv \
  --steps 50000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --edge-weight 1.5 \
  --checkpoint-every 2000 \
  --snapshot-every 5000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v35_16skin_mem \
  --device cuda
```

Evaluate normal exported metrics:

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_16skin/train.csv \
  --slotnet runs/slotnet_v35_16skin_mem/best.safetensors \
  --device cpu
```

Evaluate identity retrieval:

```bash
.venv/bin/python scripts/11_eval_slotnet_retrieval.py \
  --samples data_v35_16skin/train.csv \
  --slotnet runs/slotnet_v35_16skin_mem/best.safetensors \
  --out-json runs/slotnet_v35_16skin_mem/retrieval_train.json \
  --device cuda
```

Acceptance:

```text
top1_accuracy > 0.95
median_true_exported_pixels_mae < 0.02
no visually obvious collapse of one skin into another
```

If retrieval fails, do not rent GPU and do not jump straight to all skins. The
global-style-vector architecture is probably not separating identities enough.

## Gate 4: Same-Skin Unseen-Variant Test

After 16-skin memorization passes, split variants within the same 16 skins:

```bash
.venv/bin/python scripts/12_make_variant_split.py \
  --samples data_v35_16skin/train.csv \
  --out-dir data_v35_16skin_variant_split \
  --train-before-variant 24
```

Train with validation:

```bash
.venv/bin/python train_slotnet.py \
  --train data_v35_16skin_variant_split/train.csv \
  --val-csv data_v35_16skin_variant_split/val.csv \
  --eval-every 1000 \
  --steps 50000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --edge-weight 1.5 \
  --checkpoint-every 2000 \
  --snapshot-every 5000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v35_16skin_variant_val \
  --device cuda
```

Evaluate validation rows:

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_16skin_variant_split/val.csv \
  --slotnet runs/slotnet_v35_16skin_variant_val/best.safetensors \
  --device cpu

.venv/bin/python scripts/11_eval_slotnet_retrieval.py \
  --samples data_v35_16skin_variant_split/val.csv \
  --slotnet runs/slotnet_v35_16skin_variant_val/best.safetensors \
  --out-json runs/slotnet_v35_16skin_variant_val/retrieval_val.json \
  --device cuda
```

Acceptance:

```text
val exported_pixels_mae < 0.025-0.03
val retrieval top1_accuracy > 0.95
visual output remains sharp
```

## Do Not Rent Yet

Rent a 4090 only after all are true:

```text
1. Export dimension audit is understood and documented.
2. 3-5 diverse one-skin overfits pass.
3. 16-skin memorization passes exported metrics and retrieval.
4. 16-skin unseen-variant validation passes.
5. The training script resumes, validates, and saves best-by-val correctly.
```

First rented run should be `512 skins x 32 variants`, then `2k skins x 32`, then
full corpus. Do not start with the full corpus.

## Fallback If 16/64 Skins Fail

Do not go back to padded full-atlas training.

Keep:

```text
exact exported BMP heads
exported-pixel loss
per-file metrics
retrieval metric
```

Change only the conditioning path:

```text
global style vector -> global style + local evidence path
```

The likely next architecture is per-file heads with queries or cross-attention
into encoder feature maps, so `MAIN`, `EQMAIN`, `CBUTTONS`, and `PLEDIT` can
use local evidence from the corresponding rendered regions.
