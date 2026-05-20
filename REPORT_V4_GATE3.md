# Report: V4 Gate 3 — 16-Skin Memorization (Masked Loss)

```text
Gate 3 verdict: identity separation passes cleanly, reconstruction fails.
Useful failed Gate 3 — exactly the signal we wanted before architectural moves.
```

## Acceptance Table

| criterion                              | required          | result   | pass? |
|----------------------------------------|-------------------|----------|:-----:|
| top1 retrieval accuracy                | `> 95%`           | **1.000**| ✓     |
| median true `exported_pixels_mae`      | `< 0.02`          | 0.0390   | ✗     |
| per-file metrics reported              | yes               | yes      | ✓     |
| no obvious identity collapse           | yes               | 32/32 every skin | ✓ |
| EQMAIN hit5 tracked explicitly         | yes               | 0.567    | (tracked, weaker than Gate 2) |

The retrieval pass is unambiguous: the global style vector cleanly
separates all 16 atlases. The reconstruction fail is also unambiguous:
none of the larger trainable BMPs are within tolerance.

## Run Configuration

- Training source: `data_v4_16skin/train.csv` (16 skins × 32 variants = 512 rows, train-only split).
- Command:
  ```bash
  .venv/bin/python train_slotnet.py \
    --train data_v4_16skin/train.csv \
    --steps 50000 --batch 2 --lr 1e-4 --weight-decay 1e-4 \
    --base-channels 24 --style-dim 192 --edge-weight 1.5 \
    --checkpoint-every 2000 --snapshot-every 5000 \
    --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 2 --amp \
    --out runs/slotnet_v4_16skin_masked_mem --device cuda
  ```
- `git_commit`: `595bd09` (recorded in `runs/slotnet_v4_16skin_masked_mem/config.yaml`).
- AMP + masked-loss + Cranamp-supported-pixel mask in both loss and retrieval eval.
- Wall time: ~5h on RTX 2070 mobile, ~165 steps/min, no GPU memory pressure.
- Final per-batch loss: 0.083 (noisy single-batch; trust full-dataset metrics).

## Full-Dataset Aggregate

All metrics computed across the 512-sample training set, masked to
Cranamp-supported pixels.

| checkpoint                  | exported_pixels_mae | hit_5_255 | sobel_mae |
|----------------------------|--------------------:|----------:|----------:|
| `snapshot_step050000`       | 0.04065             | 0.6682    | 0.05852   |
| `last.safetensors`          | 0.04065             | 0.6682    | 0.05852   |
| `best.safetensors` (noisy)  | 0.04182             | 0.6626    | 0.06033   |

`last` and `snapshot_step050000` agree to 5 decimals (saved one step
apart). `best.safetensors` is marginally worse and is, as predicted in
the rolling-vs-batch discussion during training, simply a lucky earlier
minibatch — not a meaningful checkpoint to deploy.

## Per-File Breakdown (`snapshot_step050000`)

| file        |   mae    | hit_5_255 | role                                         |
|-------------|---------:|----------:|----------------------------------------------|
| playpaus    | 0.00222  | 0.9950    | small, dense, near-perfect                   |
| monoster    | 0.00911  | 0.8866    | small, mostly solved                         |
| titlebar    | 0.02642  | 0.7383    | trailing                                     |
| posbar      | 0.03139  | 0.6627    | trailing                                     |
| shufrep     | 0.03192  | 0.5967    | trailing                                     |
| pledit      | 0.05000  | 0.6441    | structural failure                           |
| balance     | 0.05002  | 0.5465    | structural failure                           |
| cbuttons    | 0.05161  | 0.5779    | structural failure                           |
| main        | 0.05675  | 0.6186    | structural failure                           |
| volume      | 0.05936  | 0.5174    | structural failure                           |
| eqmain      | 0.07830  | 0.5667    | worst — same axis as Gate 2 risk             |

**The two smallest sprites (playpaus, monoster) are essentially solved.
Every larger file fails to memorize.** Gate 2's "only EQMAIN is hard"
caveat broadens here to "any file that needs more than a couple of
samples of skin-specific information underfits".

## Retrieval (Masked)

```text
top1_accuracy        = 1.000  (512/512)
samples              = 512
target_skins         = 16
median true mae      = 0.0390
min  / max true mae  = 0.0118 / 0.0733
```

Every variant of every skin retrieved its correct target atlas. Per-skin
hits/total:

```text
a_halo_so_bright_it_bleeds       32/32
aguileramp_oldschool             32/32
blair_razor_project              32/32
cyborg                           32/32
darkside                         32/32
dragonzv30amp                    32/32
engraved4_platinum               32/32
goodgawd                         32/32
infected_fx_gray_no_transparency 32/32
minimalistic_black               32/32
rancid_amp_5                     32/32
ruki2_by_michi                   32/32
simblyblayit                     32/32
the_four_horsemen                32/32
tvxq_winamp_skins_by_roseweedy   32/32
zelda_amp_gold                   32/32
```

No identity collapse. No skin is being mistaken for another. The global
style vector is doing its job.

## Diagnosis

Maps directly onto the three branches discussed during training:

1. **High retrieval despite mediocre MAE — the model separates identities
   but the decoders underfit details.** ✓ This is the case here.
2. Low retrieval — global style vector fails to separate identities. ✗
3. Only EQMAIN/large files fail — local-evidence heads needed. Partly
   true: the failure is concentrated in large/medium files, but it is
   broader than just EQMAIN now.

The decoders have enough information to pick the right atlas but not
enough conditioning to reproduce its texture, especially for files that
carry per-skin chromatic identity (`balance`, `volume`, `main`,
`eqmain`, `cbuttons`). The global 192-dim style vector is a bottleneck.

## Comparison Against Gate 2

| metric                                   | Gate 2 best (Aguileramp 20k) | Gate 3 snap50k |
|------------------------------------------|-----------------------------:|--------------:|
| aggregate exported_pixels_mae            | 0.00577                      | 0.04065       |
| aggregate hit_5_255                      | 0.960                        | 0.668         |
| EQMAIN mae                               | 0.0141                       | 0.0783        |
| EQMAIN hit5                              | 0.845                        | 0.567         |
| PLEDIT mae                               | 0.0143                       | 0.0500        |
| PLEDIT hit5                              | 0.972                        | 0.644         |

Gate 2 (one skin) → Gate 3 (16 skins): aggregate MAE worsens ×7,
`EQMAIN` worsens ×5.5, `PLEDIT` worsens ×3.5. The slope is much steeper
than would be expected from "more diverse data" alone — it is a
capacity / conditioning problem.

## What This Result Says About Next Steps

- Do not retrain V4 with longer schedules; reconstruction is plateauing
  on conditioning capacity, not optimization budget. Curves were
  decelerating cleanly:
  ```text
   9-10k  l1 0.0798
  14-15k  l1 0.0686
  19-20k  l1 0.0599
  24-25k  l1 0.0553
  27-28k  l1 0.0525
  37-38k  l1 0.0469
  last 1k l1 0.0447
  ```
  Extrapolating the last ~5k slope to 100k still lands above 0.035.
- Do not tune file weights or edge weight. Gate 2 already showed weights
  do not help large-panel reconstruction.
- The natural next architecture matches the V4 handoff's fallback note:
  ```text
  per-file decoder heads with queries / cross-attention into encoder
  feature maps, so each file can use local evidence from the
  corresponding rendered regions.
  ```
- Keep the masked exported-BMP loss and the support profile. They are
  the right targets.
- Do not return to padded full-atlas training.
- No 4090 rental.

## Eval Artifacts (committed under `reports/v4_gate3_candidates/`)

- `snap50k_eval.json`, `last_eval.json`, `best_eval.json` — full-dataset 09_eval per checkpoint.
- `retrieval_snap50k.json` — masked top-1 retrieval (1.000 / 512).
- (No new exported `.wsz` for this gate; identity-separation evidence is the deliverable.)
