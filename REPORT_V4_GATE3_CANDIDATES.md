# Report: V4 Gate 3 — 16-Skin Candidate Selection

Dataset is built (`data_v4_16skin/`, 512 samples) but training is held
until the candidate list is reviewed and approved.

## How the 16 were chosen

`scripts/14_pick_16skin.py` profiles each candidate skin on five
normalized features from its `MAIN.bmp`:

```text
L_mean         brightness
L_std          contrast
saturation     mean (max-min)/max across channels
log_palette    log10 of unique-color estimate
edge_density   mean absolute gradient
```

The three Gate 2 anchors (darkside, Aguileramp_-_OldSchool, Zelda_Amp_Gold)
are seeded; the remaining 13 are selected by farthest-point sampling in
the normalized feature space. A viability filter excludes degenerate
sources (`palette_estimate < 30` or `L_std < 20`) so the picks stay
within plausibly trainable skins. Pool: 500 random non-anchor skins,
seed `20260520`.

## The 16

| # | skin | L | std | sat | palette | edge | role |
|--:|------|---:|----:|----:|--------:|-----:|------|
|  0 | `darkside.wsz` | 32.4 | 50.7 | 0.10 |  446 | 0.04 | Gate 2 anchor |
|  1 | `Aguileramp_-_OldSchool.wsz` | 195.0 | 86.9 | 0.26 |  463 | 0.06 | Gate 2 anchor |
|  2 | `Zelda_Amp_Gold.wsz` | 56.2 | 54.8 | 0.89 |  713 | 0.05 | Gate 2 anchor |
|  3 | `GoodGawd.wsz` | 140.9 | 123.9 | 1.00 |   85 | 0.09 | extreme saturation + contrast |
|  4 | `The_Four_Horsemen.wsz` | 93.7 | 88.3 | 0.41 |  195 | 0.17 | busy texture / high edge density |
|  5 | `engraved4_platinum.wsz` | 228.0 | 26.6 | 0.09 |   95 | 0.01 | bright flat-UI / low contrast |
|  6 | `minimalistic_black.wsz` | 106.1 | 68.9 | 0.47 |   30 | 0.02 | low-palette mid-saturation |
|  7 | `a_halo_so_bright_it_bleeds.wsz` | 93.7 | 99.2 | 0.60 | 3629 | 0.12 | photographic-saturated |
|  8 | `Cyborg.wsz` | 153.3 | 46.3 | 0.09 | 4090 | 0.04 | muted photographic |
|  9 | `Rancid_Amp_5.wsz` | 73.6 | 105.6 | 0.05 |  361 | 0.09 | dark, very-high-contrast greyscale |
| 10 | `tvxq_winamp_skins_by_roseweedy.wsz` | 191.8 | 82.5 | 0.07 | 1075 | 0.15 | bright B&W photographic |
| 11 | `simblyblayit.wsz` | 160.2 | 96.2 | 0.78 |  367 | 0.03 | bright saturated flat-UI |
| 12 | `Infected FX - Gray No Transparency.wsz` | 129.8 | 55.6 | 0.00 |  147 | 0.08 | fully greyscale |
| 13 | `Ruki2 by michi.wsz` | 100.0 | 73.1 | 0.58 | 4572 | 0.04 | photographic-saturated flat-UI |
| 14 | `DragonZV30amp.wsz` | 74.0 | 75.5 | 0.49 |  213 | 0.09 | dark mid-saturation |
| 15 | `blair_razor_project.wsz` | 61.3 | 37.6 | 0.13 | 3693 | 0.08 | dark photographic |

Coverage spans the requested axes:

- **brightness**: dark `kago/darkside (4.9, 32)` → bright `engraved4_platinum (228)`.
- **contrast**: low `engraved4_platinum (27)` → high `GoodGawd (124)`.
- **saturation**: zero `Infected FX (0.00)` → max `GoodGawd (1.00)`.
- **palette**: small `minimalistic_black (30)` → photographic `Ruki2 (4572)`.
- **edge density**: flat `Cyborg (0.04)` / `engraved4 (0.01)` → busy `The_Four_Horsemen (0.17)`.

Contact sheet: `reports/v4_gate3_candidates/main_contact_sheet.png`
(4×4 grid of each skin's `MAIN.bmp` rendered at canonical 275×115).

## Pack / Render status

```text
packed 16/16 skin source(s); wrote data_v4_16skin/valid_skins.csv
rendering 512 sample(s) across 4 worker(s)...
done. 512 ok, 0 failed
wrote splits for 512 sample(s) in data_v4_16skin
checked 512 dataset sample(s)
```

No warnings. Dataset layout:

```text
data_v4_16skin/
├── atlases/         (16 atlas PNGs, one per skin)
├── views/           (512 rendered PNGs, 32 variants × 16 skins)
├── valid_skins.csv
├── train.csv        (512 rows; all rows train per Gate 3 contract)
├── val.csv          (empty)
├── test.csv         (empty)
```

Gate 3 is a memorization test, so the splits are `--train 1.0 --val 0.0
--test 0.0` exactly per `HANDOFF_V4.md`.

## Reproducibility

```bash
.venv/bin/python scripts/14_pick_16skin.py
# → reports/v4_gate3_candidates/picks.{json,txt}

mkdir -p /tmp/uigen_v4_16skin_raw
while read name; do
  ln -sf "/home/s/develop/projects/uigen/skins_raw/$name" "/tmp/uigen_v4_16skin_raw/$name"
done < reports/v4_gate3_candidates/picks.txt

.venv/bin/python scripts/01_pack_skins.py \
  --skins-raw /tmp/uigen_v4_16skin_raw \
  --atlas-profile configs/atlas_train_v1.json \
  --out data_v4_16skin

.venv/bin/python scripts/02_render_dataset.py \
  --valid-skins data_v4_16skin/valid_skins.csv \
  --out data_v4_16skin --variants 32 --workers 4 \
  --cranamp-cli ./cranamp_cli/cranamp-cli

.venv/bin/python scripts/03_make_splits.py \
  --data data_v4_16skin --valid-skins data_v4_16skin/valid_skins.csv \
  --train 1.0 --val 0.0 --test 0.0

.venv/bin/python scripts/04_check_dataset.py --data data_v4_16skin
```

## Planned training (not started)

```bash
.venv/bin/python train_slotnet.py \
  --train data_v4_16skin/train.csv \
  --steps 50000 \
  --batch 2 --lr 1e-4 --weight-decay 1e-4 \
  --base-channels 24 --style-dim 192 --edge-weight 1.5 \
  --checkpoint-every 2000 --snapshot-every 5000 \
  --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 2 --amp \
  --out runs/slotnet_v4_16skin_masked_mem \
  --device cuda
```

Planned acceptance (per `HANDOFF_V4.md` Gate 3 + `toclaude.md`):

```text
top1 retrieval accuracy > 95%   (scripts/11_eval_slotnet_retrieval.py)
median exported_pixels_mae < 0.02
per-file metrics reported
no visually obvious skin identity collapse
EQMAIN hit5 tracked explicitly
```

## Awaiting Approval

Per `toclaude.md`: do not start 50k training until the candidate list is
reviewed. Likely review questions:
- any obvious "too easy" skin to swap (e.g. `engraved4_platinum`'s flat-UI)?
- coverage gaps to cover with a manual swap?
- any of the photographic skins that visually overlap (e.g. `Ruki2` vs
  `blair_razor_project` both photographic)?

After review, the same trainer command above can be launched.
