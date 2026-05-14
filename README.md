# Cranamp Atlas AI

This repo implements the early, skin-corpus-independent pieces of the Cranamp render-to-atlas roadmap.

Current scope:

- scan raw Winamp skin folders or `.wsz` archives
- pack skin BMPs into a fixed `1024x1024` RGB atlas
- write atlas masks, per-slot weights, metadata, and `valid_skins.csv`
- export a packed atlas back to a classic skin folder and `skin.wsz`
- keep configs grounded in the local Cranamp implementation and the supplied first skin

The first default skin is committed under `assets/default_skin/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scan And Pack

Put raw skins in `skins_raw/` as either folders or `.wsz` files, then run:

```bash
python scripts/00_scan_skins.py --skins-raw skins_raw --out data_v0/skin_scan.csv
python scripts/01_pack_skins.py \
  --skins-raw skins_raw \
  --atlas-profile configs/atlas_v1.json \
  --export-profile configs/export_profile_classic.json \
  --default-skin assets/default_skin \
  --out data_v0
```

For a one-skin smoke test before collecting a corpus, point `--skins-raw` at a directory containing the supplied skin folder.

## Export

```bash
python scripts/05_export_atlas_to_skin.py \
  --atlas data_v0/atlases/<skin_id>.png \
  --atlas-profile configs/atlas_v1.json \
  --export-profile configs/export_profile_classic.json \
  --default-skin assets/default_skin \
  --out out_skin
```

## One-Skin Smoke Path

This runs everything that can execute before a large skin corpus exists:

```bash
python scripts/01_pack_skins.py --skins-raw assets --out /tmp/uigen_data_v0
python scripts/02_render_dataset.py --valid-skins /tmp/uigen_data_v0/valid_skins.csv --out /tmp/uigen_data_v0 --variants 2 --cranamp-cli ./cranamp_cli/cranamp-cli
python scripts/03_make_splits.py --data /tmp/uigen_data_v0 --valid-skins /tmp/uigen_data_v0/valid_skins.csv
python scripts/04_check_dataset.py --data /tmp/uigen_data_v0 --debug-out /tmp/uigen_data_v0/debug/contact.png
python train_geonet.py --train /tmp/uigen_data_v0/train.csv --max-steps 1 --base-channels 4 --fpn-channels 8 --out /tmp/uigen_runs/geonet --device cpu
python train_slotnet.py --train /tmp/uigen_data_v0/train.csv --slot PLAYPAUS --steps 1 --out /tmp/uigen_runs/slotnet --device cpu
python eval_pipeline.py --samples /tmp/uigen_data_v0/train.csv --out /tmp/uigen_eval --limit 1
```

The output is only a pipeline smoke test. Real quality is blocked on many skins and GPU training.

## Cranamp

The local Cranamp checkout used for inspection is expected at:

```bash
export CRANAMP_REPO=/home/s/develop/projects/cranamp
```

Renderer-driven dataset generation is available through:

```bash
./cranamp_cli/cranamp-cli render-random --help
```

The current renderer is the first deterministic CLI compositor in the copied
Cranamp fork. It emits views, rect labels, state labels, params, and visible
atlas masks; deeper parity with the interactive Cranamp renderer is tracked in
the roadmap.
