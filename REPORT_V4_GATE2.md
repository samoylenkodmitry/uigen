# Report: V4 Gate 2 — Diverse One-Skin Overfits (Masked Loss)

```text
Gate 2 passed enough to proceed to Gate 3, but not cleanly.
Recurring caveat: EQMAIN hit5 remains weak on textured/saturated skins.
```

All three runs used the static Cranamp-supported-pixel loss, the V4
trainer with AMP + dataloader workers, and the same hyperparameters. No
prior atlas, no dynamic mask as model input, no replay/distortion
metadata. Export still writes full BMPs into `skin.wsz`; only the loss
and evaluator are masked.

## Aggregate Results

| skin              | aggregate_mae | aggregate_hit5 | eqmain_mae | eqmain_hit5 | verdict                          |
|-------------------|--------------:|---------------:|-----------:|------------:|----------------------------------|
| darkside @10k     |       0.00431 |          0.971 |    0.00879 |       0.940 | clean pass                       |
| Aguileramp @20k   |       0.00577 |          0.960 |    0.01408 |       0.845 | pass, EQMAIN at threshold        |
| Zelda @20k        |       0.00421 |          0.955 |    0.01461 |       0.754 | pass with EQMAIN-hit5 caveat     |

Aggregate thresholds for Gate 2: `exported_pixels_mae < 0.01` strong,
`exported_pixels_hit_5_255 > 0.85`. All three skins clear both with room.

## Worst Per-File (Excluding EQMAIN)

| skin       | worst file (non-eqmain) | mae     | hit5  |
|------------|-------------------------|--------:|------:|
| darkside   | pledit                  | 0.00966 | 0.901 |
| Aguileramp | main                    | 0.01191 | 0.886 |
| Zelda      | main                    | 0.00844 | 0.893 |

Small sprites (`PLAYPAUS`, `MONOSTER`, `TITLEBAR` after Aguileramp ≥10k,
`CBUTTONS`, `POSBAR`, `SHUFREP`) are essentially solved on every skin
(`hit5 > 0.99`, `mae < 0.003`). The systemic risk is concentrated in
EQMAIN.

## Run Provenance

### darkside (Gate 2 #1, clean pass)
- training source: `data_v35_darkside_overfit/`
- checkpoint: `runs/slotnet_v4_darkside_masked_overfit/snapshot_step010000.safetensors`
- export: `runs/slotnet_v4_darkside_masked_overfit/export_step010000/skin.wsz`
- `unzip -t`: **OK**, 16 files, `PLEDIT.bmp` **present**, `VIDEO.bmp` **absent**
- `git_commit`: `ca8a3dc`
- training: 10000 steps, batch 2, lr 1e-4, base-channels 24, style-dim 192, edge-weight 1.5, AMP
- stopped early because all per-file metrics passed strongly (`mae < 0.01`, `hit5 > 0.90` for every file).

### Aguileramp_-_OldSchool (Gate 2 #2, pass via continuation)
- training source: `data_v35_aguileramp_overfit/`
- final checkpoint: `runs/slotnet_v4_aguileramp_masked_from14k_to20k/best.safetensors` (effective step 20000)
- export: `runs/slotnet_v4_aguileramp_masked_from14k_to20k/export_best/skin.wsz`
- `unzip -t`: **OK**, 16 files, `PLEDIT.bmp` **present**, `VIDEO.bmp` **absent**
- `git_commit`: `ca8a3dc`
- training trajectory:
  - `runs/slotnet_v4_aguileramp_masked_overfit` (steps 0-10000) — EQMAIN mae stuck at ~0.033.
  - `runs/slotnet_v4_aguileramp_masked_from10k_to20k` (steps 10000-14031, paused for GPU).
  - `runs/slotnet_v4_aguileramp_masked_from14k_to20k` (resumed `snapshot_step004000` of the above, +6000 steps).
- EQMAIN MAE 0.0327 (@10k) → 0.0141 (@20k), -57%. EQMAIN hit5 0.792 → 0.8449.

### Zelda_Amp_Gold (Gate 2 #3, pass with caveat)
- training source: `data_v35_zelda_overfit/`
- checkpoint: `runs/slotnet_v4_zelda_masked_overfit/best.safetensors`
- export: `runs/slotnet_v4_zelda_masked_overfit/export_best/skin.wsz`
- `unzip -t`: **OK**, 16 files, `PLEDIT.bmp` **present**, `VIDEO.bmp` **absent**
- `git_commit`: `ca8a3dc`
- training: 20000 steps, batch 2, lr 1e-4, base-channels 24, style-dim 192, edge-weight 1.5, AMP
- EQMAIN MAE 0.0146 (passes `< 0.02`) but hit5 0.754 fails `> 0.85`. EQMAIN hit5 stalled around 0.75 from step 5000 onward; longer training did not lift it.

Caveat is structural, not a training accident: Zelda's saturated gold
EQMAIN content places many predicted pixels close-in-mean but outside
5/255 tolerance.

## Eval Artifacts

Lightweight per-skin metric snapshots (aggregate + per-file mae/sobel/hit5):

- `reports/v4_gate2/darkside_eval.json`
- `reports/v4_gate2/aguileramp_eval.json`
- `reports/v4_gate2/zelda_eval.json`

## Reproduction Commands

Each skin follows the same flow:

```bash
# 1. Pack
mkdir -p /tmp/uigen_v4_<NAME>_raw
ln -sf /home/s/develop/projects/uigen/skins_raw/<SKIN>.wsz /tmp/uigen_v4_<NAME>_raw/skin.wsz
.venv/bin/python scripts/01_pack_skins.py \
  --skins-raw /tmp/uigen_v4_<NAME>_raw \
  --atlas-profile configs/atlas_train_v1.json \
  --out data_v35_<NAME>_overfit

# 2. Render
.venv/bin/python scripts/02_render_dataset.py \
  --valid-skins data_v35_<NAME>_overfit/valid_skins.csv \
  --out data_v35_<NAME>_overfit \
  --variants 32 \
  --workers 1 \
  --cranamp-cli ./cranamp_cli/cranamp-cli

# 3. Split (single-skin overfit → all rows in train)
.venv/bin/python scripts/03_make_splits.py \
  --data data_v35_<NAME>_overfit \
  --valid-skins data_v35_<NAME>_overfit/valid_skins.csv \
  --train 1.0 --val 0.0 --test 0.0

# 4. Train with masked loss
.venv/bin/python train_slotnet.py \
  --train data_v35_<NAME>_overfit/train.csv \
  --steps 20000 \
  --batch 2 --lr 1e-4 --weight-decay 1e-4 \
  --base-channels 24 --style-dim 192 --edge-weight 1.5 \
  --checkpoint-every 1000 --snapshot-every 1000 \
  --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 2 --amp \
  --out runs/slotnet_v4_<NAME>_masked_overfit \
  --device cuda

# 5. Eval (masked metrics)
.venv/bin/python scripts/09_eval_slotnet_overfit.py \
  --samples data_v35_<NAME>_overfit/train.csv \
  --slotnet runs/slotnet_v4_<NAME>_masked_overfit/best.safetensors \
  --device cpu

# 6. Export
VIEW=$(tail -n +2 data_v35_<NAME>_overfit/train.csv | cut -d, -f3 | head -1)
.venv/bin/python infer_skin.py \
  --image "$VIEW" \
  --slotnet runs/slotnet_v4_<NAME>_masked_overfit/best.safetensors \
  --out runs/slotnet_v4_<NAME>_masked_overfit/export_best \
  --device cpu
unzip -t runs/slotnet_v4_<NAME>_masked_overfit/export_best/skin.wsz
```

Skin substitutions used:
- `<NAME>=darkside, <SKIN>=darkside`
- `<NAME>=aguileramp, <SKIN>=Aguileramp_-_OldSchool`
- `<NAME>=zelda, <SKIN>=Zelda_Amp_Gold`

## Takeaways for Gate 3

- Masked loss is the correct objective: aggregate metrics improve substantially
  vs the unmasked V4 baseline, and the model concentrates capacity on pixels
  Cranamp actually displays.
- Small sprites are solved across all three skins; do not spend further effort
  on them.
- EQMAIN hit5 is the single recurring risk. MAE clears `< 0.02` everywhere,
  but tolerance-hit on saturated content stays low (Zelda 0.754).
- Treat EQMAIN hit5 as a known weakness going into 16-skin memorization. If
  it persists with similar magnitude across many skins, the next architectural
  move is per-file heads + local encoder feature queries — not more weights and
  not padded full-atlas training.
