# Handoff For Claude: BlueCurve SlotNet V3.4 Overfit

Goal: prove the current simplified pipeline can overfit one supported skin.
Every distorted input view should map back to the same clean expected
BlueCurve atlas.

## Current Code Contract

- Model: `models/slotnet_v34.py`
- Trainer: `train_slotnet.py`
- Loss: plain RGB L1 in `models/losses.py`
- Direct MAE evaluator: `scripts/09_eval_slotnet_overfit.py`
- Training target profile: `configs/atlas_train_v1.json`
- Export profile: `configs/export_profile_classic.json`
- Source skin: `skins_raw/BlueCurve_Winamp.wsz`

Do not use pre-V3.4 datasets, old checkpoints, prior atlases, observed heads,
visible masks, rect/state side channels, replay params, debug contact sheets,
or preview renders as training inputs.

## Invariants

- Input to SlotNet is only the rendered RGB PNG tensor.
- Output from SlotNet is only a 3-channel RGB atlas tensor.
- Target is only the expected RGB atlas PNG.
- Distortions exist only in the rendered input PNG pixels.
- Dataset CSV rows should contain only:
  `skin_id, variant_id, view_png, atlas_png, meta_json`.
- The BlueCurve overfit dataset should have many distorted views, but all rows
  must point to the same expected atlas PNG.
- Exported skin folder should stay focused on skin files plus `atlas.png` and
  `skin.wsz`; do not write load-smoke preview PNGs there.

## Build Fresh BlueCurve Dataset

Use a fresh dataset root; do not train from the one-step smoke folder.

```bash
mkdir -p /tmp/uigen_v34_bluecurve_raw
ln -sf /home/s/develop/projects/uigen/skins_raw/BlueCurve_Winamp.wsz /tmp/uigen_v34_bluecurve_raw/skin.wsz

.venv/bin/python scripts/01_pack_skins.py \
  --skins-raw /tmp/uigen_v34_bluecurve_raw \
  --atlas-profile configs/atlas_train_v1.json \
  --out data_v34_bluecurve_overfit

.venv/bin/python scripts/02_render_dataset.py \
  --valid-skins data_v34_bluecurve_overfit/valid_skins.csv \
  --out data_v34_bluecurve_overfit \
  --variants 32 \
  --workers 1 \
  --cranamp-cli ./cranamp_cli/cranamp-cli

.venv/bin/python scripts/03_make_splits.py \
  --data data_v34_bluecurve_overfit \
  --valid-skins data_v34_bluecurve_overfit/valid_skins.csv

.venv/bin/python scripts/04_check_dataset.py \
  --data data_v34_bluecurve_overfit
```

Expected dataset shape:

```bash
find data_v34_bluecurve_overfit -maxdepth 2 -type f | sort
```

There should be `views/`, `atlases/`, CSVs, and metadata only. No `debug/`,
`params/`, `rects/`, `states/`, `visible_masks/`, or replay JSON files.

## Train One-Skin Overfit

Start with base 24. If CUDA memory fails, drop to base 16, but do not change the
data contract.

```bash
.venv/bin/python train_slotnet.py \
  --train data_v34_bluecurve_overfit/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --out runs/slotnet_v34_bluecurve_overfit \
  --device cuda
```

## Measure Overfit

Use direct atlas MAE, not visual familiarity.

```bash
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v34_bluecurve_overfit/train.csv \
  --slotnet runs/slotnet_v34_bluecurve_overfit/last.safetensors \
  --device cpu
```

Acceptance target for this one-skin overfit:

- `rgb_mae < 0.02`: acceptable first pass.
- `rgb_mae < 0.01`: strong evidence the model can memorize the expected atlas.
- If MAE plateaus above `0.02`, stop and debug the simple path before adding
  capacity, priors, masks, or auxiliary heads.

## Export A Checkpoint

Export one distorted input view through the trained checkpoint:

```bash
VIEW=$(tail -n +2 data_v34_bluecurve_overfit/train.csv | cut -d, -f3 | head -1)

.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v34_bluecurve_overfit/last.safetensors \
  --out runs/slotnet_v34_bluecurve_overfit/export_last \
  --device cpu

unzip -t runs/slotnet_v34_bluecurve_overfit/export_last/skin.wsz
```

Expected export folder: generated BMPs, `atlas.png`, and `skin.wsz`. It should
include `PLEDIT.bmp` and should not include `VIDEO.bmp`.

Load checks may render to `/tmp`, but do not leave random-render preview PNGs in
the export folder because they look intentionally distorted and confuse review.
