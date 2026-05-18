# Cranamp Atlas AI

This repo trains a direct Cranamp skin-atlas model.

Current V3.5 contract:

```text
input rendered PNG -> predicted exported BMP tensors -> expected BMP pixels
```

Unsupported components are removed from the training atlas. The model does not
use a default-skin prior, observed auxiliary head, dynamic masks, special-color
head, or distortion metadata.

## Active Files

- Training atlas profile: `configs/atlas_train_v1.json`
- Export profile: `configs/export_profile_classic.json`
- SlotNet model: `models/slotnet_v35.py`
- SlotNet trainer: `train_slotnet.py`
- Inference/export: `infer_skin.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build A Fresh V3.5 Dataset

Put raw skins in `skins_raw/` as folders or `.wsz` archives, then run:

```bash
python scripts/00_scan_skins.py --skins-raw skins_raw --out data_v35/skin_scan.csv
python scripts/01_pack_skins.py --skins-raw skins_raw --out data_v35
python scripts/02_render_dataset.py --valid-skins data_v35/valid_skins.csv --out data_v35 --variants 16 --cranamp-cli ./cranamp_cli/cranamp-cli
python scripts/03_make_splits.py --data data_v35 --valid-skins data_v35/valid_skins.csv
python scripts/04_check_dataset.py --data data_v35
```

The split CSVs contain only the V3.5 training contract: rendered input PNG,
expected RGB atlas PNG, and packed-skin metadata. The random render parameters
are not saved. Distortion is intentionally present only in the input PNG pixels.
Rendered inputs include the playlist because Cranamp supports and displays it.

## Train V3.5

```bash
python train_slotnet.py \
  --train data_v35/train.csv \
  --steps 20000 \
  --batch 2 \
  --base-channels 24 \
  --edge-weight 1.5 \
  --out runs/slotnet_v35_1skin \
  --device cuda
```

For a one-skin overfit smoke, generate a one-skin `data_v35` first and train
with `--limit-rows` if needed. Pre-V3.5 datasets, checkpoints, and eval folders
are intentionally not kept in the repo workspace.

## Inference

```bash
python infer_skin.py \
  --image path/to/mockup.png \
  --slotnet runs/slotnet_v35_1skin/best.safetensors \
  --out out_skin
```

This writes `out_skin/atlas.png` and `out_skin/skin.wsz`.
