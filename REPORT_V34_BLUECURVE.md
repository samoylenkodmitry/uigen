# Report: SlotNet V3.4 BlueCurve 1-Skin Overfit

Result: **passed** the "strong" acceptance from `HANDOFF_V34.md`.

## Setup

- Source skin: `skins_raw/BlueCurve_Winamp.wsz` (one skin, 32 distorted views,
  all rows point to the same expected atlas).
- Dataset root: `data_v34_bluecurve_overfit/` (32 train rows, plus val/test
  splits over the same skin).
- Model: `models/slotnet_v34.py` (pure RGB L1, no prior, no aux head, no per-slot
  weights).
- Loss: plain RGB L1 in `models/losses.py`.
- Training target profile: `configs/atlas_train_v1.json`.
- Export profile: `configs/export_profile_classic.json`.
- Dataset CSV rows carry only `skin_id, variant_id, view_png, atlas_png, meta_json`.

## Training

```
.venv/bin/python train_slotnet.py \
  --train data_v34_bluecurve_overfit/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 16 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --out runs/slotnet_v34_bluecurve_overfit \
  --device cuda
```

HANDOFF specified `--base-channels 24`; it OOM'd on a 7.6 GiB GPU at
batch=2, so per the HANDOFF fallback ("If CUDA memory fails, drop to base 16")
the run was restarted at base=16. Data contract and other hyperparameters
unchanged.

Runtime: ~2h45m wall time on RTX 2070 Mobile, ~115 steps/min.

## Training Loss Trajectory

Single-step rgb_mae at each 1000-step milestone:

| step  | rgb_mae |
|------:|--------:|
|     0 | 0.4520  |
|  1000 | 0.0978  |
|  2000 | 0.0521  |
|  3000 | 0.0355  |
|  4000 | 0.0257  |
|  5000 | 0.0210  |
|  6000 | 0.0185  |
|  7000 | 0.0163  |
|  8000 | 0.0149  |
|  9000 | 0.0141  |
| 10000 | 0.0131  |
| 11000 | 0.0126  |
| 12000 | 0.0121  |
| 13000 | 0.0117  |
| 14000 | 0.0113  |
| 15000 | 0.0111  |
| 16000 | 0.0109  |
| 17000 | 0.0106  |
| 18000 | 0.0104  |
| 19000 | 0.0102  |
| 19999 | 0.0100  |

Monotonic, no plateau above 0.02, no obvious regression.

## Direct Atlas MAE (eval)

`scripts/09_eval_slotnet_overfit.py` was run on the full 32-variant train set
against `last.safetensors`:

```
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v34_bluecurve_overfit/train.csv \
  --slotnet runs/slotnet_v34_bluecurve_overfit/last.safetensors \
  --device cpu
```

- aggregate `rgb_mae`: **0.009983** over 32 samples.
- per-variant range: ~0.0099 to ~0.0101 (tight, no outliers).

Acceptance from `HANDOFF_V34.md`:
- `rgb_mae < 0.02`: acceptable first pass.
- `rgb_mae < 0.01`: strong evidence the model can memorize the expected atlas.

The aggregate is below 0.01, so this lands in the "strong" band.

## Export

```
VIEW=$(tail -n +2 data_v34_bluecurve_overfit/train.csv | cut -d, -f3 | head -1)

.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v34_bluecurve_overfit/last.safetensors \
  --out runs/slotnet_v34_bluecurve_overfit/export_last \
  --device cpu
```

`runs/slotnet_v34_bluecurve_overfit/export_last/skin.wsz` is a clean zip of 16
entries. Atlas-derived BMPs from this run: BALANCE, CBUTTONS, EQMAIN, MAIN,
MONOSTER, PLAYPAUS, PLEDIT, POSBAR, SHUFREP, TITLEBAR, VOLUME. Default-skin
passthrough BMPs/TXTs: NUMBERS, PLEDIT.TXT, README.txt, TEXT, VISCOLOR.TXT.

Required by HANDOFF: PLEDIT.bmp present (yes), VIDEO.bmp absent (yes).
`unzip -t` reports no errors.

## Takeaways

- The simplified V3.4 contract (full-image input, direct atlas output, plain L1)
  is capable of memorizing a single skin's atlas end-to-end, with no prior,
  visible mask, rect/state side channel, or auxiliary head.
- Auto-detection of `base_channels` in `infer_skin.py` correctly loaded a base=16
  checkpoint without manual flag changes.
- 32 variants of one skin at base=16 are enough to drive the loss below 0.01;
  next step is to confirm the path generalizes to a small multi-skin set
  before adding capacity.
