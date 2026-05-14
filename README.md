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

## Cranamp

The local Cranamp checkout used for inspection is expected at:

```bash
export CRANAMP_REPO=/home/s/develop/projects/cranamp
```

Renderer-driven dataset generation still depends on adding or wrapping `cranamp-cli`; see `cranamp_cli/README.md`.
