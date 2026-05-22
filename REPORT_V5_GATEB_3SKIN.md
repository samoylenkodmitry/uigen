# SlotNet V5 Gate B — Three-Skin Overfit

Date: 2026-05-21

## Conclusion

V5 passes Gate B from scratch on a three-skin combined dataset
(BlueCurve + DarkSide + Zelda).

```text
retrieval top1                 1.000   pass
exported_pixels_mae            0.00554 pass (<0.015)
exported_pixels_hit_5_255      0.971   pass (>0.90)
exported_pixels_sobel_mae      0.00956
.wsz export clean per skin     yes (all 3 skins)
PLEDIT.bmp present per skin    yes
VIDEO.bmp absent per skin      yes
```

Every per-skin aggregate is above 0.95 hit5; the recurring per-file weakness
from V4 Gate 2 (EQMAIN hit5 on Zelda; balance on Zelda) is still visible but
the rest of each skin is essentially solved.

Gate B is the precondition for Gate C. V5 is cleared to attempt the 16-skin
gate that V4 failed on reconstruction.

## Acceptance vs. Result

Acceptance criteria (per user instruction):

```text
retrieval top1 = 1.0
median/exported MAE < 0.015 - 0.02
per-file metrics reported
.wsz export clean for each selected skin
```

Selected checkpoint `snapshot_step020000.safetensors`:

```text
retrieval top1_accuracy                 1.000      (96 / 96 samples)
retrieval mean_best_exported_mae        0.00554
retrieval median_best_exported_mae      0.00409
masked exported_pixels_mae              0.00554
masked exported_pixels_hit_5_255        0.971
masked exported_pixels_sobel_mae        0.00956
full_atlas_mae                          0.03060    (informational only)
```

## Dataset

```text
data_v5_gateB_3skin/train.csv             96 rows (32 per skin)
  skin_180ffb08  BlueCurve_(deepfried)    (V35 baseline positive)
  skin_127876f0  DarkSide                  (V4 Gate 2 clean pass)
  skin_3cc38af4  Zelda_Amp_Gold            (V4 Gate 2 EQMAIN-weak)
```

Each skin contributes 32 distorted rendered input views against one clean
target atlas. Train CSV built by concatenating the three V3.5 overfit splits.

## Run

Run directory:

```text
runs/slotnet_v5_3skin_gateB/
```

Trained from scratch with `--model-version 50` (no resume from V5 Gate A
BlueCurve checkpoint). Command (same hyperparameters as Gate A):

```bash
.venv/bin/python train_slotnet.py \
  --model-version 50 \
  --train data_v5_gateB_3skin/train.csv \
  --steps 20000 \
  --batch 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --base-channels 24 \
  --style-dim 192 \
  --head-channels 96 \
  --attn-dim 128 \
  --attention-heads 4 \
  --cross-attention-layers 1 \
  --file-embedding-dim 32 \
  --edge-weight 1.5 \
  --checkpoint-every 1000 \
  --snapshot-every 1000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v5_3skin_gateB \
  --device cuda
```

Wall time: ~2h12m on a single RTX 2070 (8 GB), AMP, batch 2.

Final training loss: 0.01421 (single-batch noisy; selection on full-dataset
eval below).

## Aggregate Per-File Metrics (selected checkpoint, step 20000)

```text
file        MAE        hit_5_255   sobel_MAE
balance     0.020640   0.953077    0.044230
cbuttons    0.002762   0.994111    0.004195
eqmain      0.011669   0.864943    0.018322
main        0.007831   0.909127    0.010954
monoster    0.002017   0.999504    0.002618
playpaus    0.000515   1.000000    0.000739
pledit      0.006012   0.974149    0.010382
posbar      0.001956   0.999902    0.003013
shufrep     0.002132   0.999025    0.003039
titlebar    0.001868   0.998665    0.002475
volume      0.003487   0.989083    0.005199
```

Small sprites are essentially solved (`monoster`, `playpaus`, `posbar`,
`shufrep`, `titlebar`, `cbuttons` all hit5 > 0.99 and MAE < 0.003).

Two soft files at the aggregate level:

- `balance` MAE is 0.021 (driven entirely by Zelda; see per-skin below).
- `eqmain` hit5 is 0.865 — under 0.90 but consistent with V4 Gate 2 where
  EQMAIN is the recurring caveat on saturated/textured skins.

## Per-Skin Aggregate

```text
skin                       MAE       hit5      sobel_MAE
skin_127876f0  DarkSide    0.00388   0.9813    0.00672
skin_180ffb08  BlueCurve   0.00409   0.9743    0.00530
skin_3cc38af4  Zelda       0.00864   0.9576    0.01666
```

Zelda is the relatively hardest skin, matching the V4 Gate 2 pattern.

## Per-Skin Per-File

```text
DarkSide (skin_127876f0)
  balance    mae=0.00284  hit5=0.99892
  cbuttons   mae=0.00303  hit5=0.98919
  eqmain     mae=0.00899  hit5=0.93545
  main       mae=0.00677  hit5=0.91957
  monoster   mae=0.00273  hit5=0.99851
  playpaus   mae=0.00005  hit5=1.00000
  pledit     mae=0.00858  hit5=0.96667
  posbar     mae=0.00235  hit5=0.99983
  shufrep    mae=0.00242  hit5=0.99894
  titlebar   mae=0.00154  hit5=0.99895
  volume     mae=0.00336  hit5=0.98848

BlueCurve (skin_180ffb08)
  balance    mae=0.00493  hit5=0.97142
  cbuttons   mae=0.00245  hit5=0.99945
  eqmain     mae=0.01341  hit5=0.87914
  main       mae=0.00835  hit5=0.90349
  monoster   mae=0.00132  hit5=1.00000
  playpaus   mae=0.00077  hit5=1.00000
  pledit     mae=0.00441  hit5=0.97925
  posbar     mae=0.00128  hit5=1.00000
  shufrep    mae=0.00214  hit5=0.99930
  titlebar   mae=0.00184  hit5=0.99985
  volume     mae=0.00406  hit5=0.98507

Zelda (skin_3cc38af4)
  balance    mae=0.05415  hit5=0.88889    <- outlier
  cbuttons   mae=0.00280  hit5=0.99369
  eqmain     mae=0.01261  hit5=0.78024    <- recurring EQMAIN caveat
  main       mae=0.00837  hit5=0.90432
  monoster   mae=0.00201  hit5=1.00000
  playpaus   mae=0.00072  hit5=1.00000
  pledit     mae=0.00504  hit5=0.97653
  posbar     mae=0.00224  hit5=0.99988
  shufrep    mae=0.00183  hit5=0.99884
  titlebar   mae=0.00222  hit5=0.99719
  volume     mae=0.00304  hit5=0.99370
```

The aggregate `balance` MAE (0.021) is dominated entirely by Zelda's BALANCE
panel (0.054); BlueCurve and DarkSide BALANCE are well-fit (MAE 0.005 and
0.003 respectively). The aggregate `eqmain` hit5 weakness comes from Zelda
(0.78) and BlueCurve (0.88); DarkSide EQMAIN sits at 0.94.

## Retrieval

`scripts/11_eval_slotnet_retrieval.py` evaluates each input view against the
three target skins by minimum exported-pixel MAE.

```text
samples         96
target_skins    3
top1_accuracy   1.0    (96 / 96 correct)
```

Per-skin retrieval best-MAE distribution is tight and well-separated:

```text
BlueCurve views   best MAE ~0.00409  (predicted skin_180ffb08, correct)
DarkSide views    best MAE ~0.00388  (predicted skin_127876f0, correct)
Zelda views       best MAE ~0.00864  (predicted skin_3cc38af4, correct)
```

No misclassifications. The 2-3x gap between Zelda and BlueCurve/DarkSide
reconstruction quality does not cause any identity confusion.

## Learning Curve

CPU evals at 5k and 10k on the same combined CSV:

```text
metric              5k        10k       20k
exported_mae        0.01337   0.00956   0.00554
hit_5_255           0.857     0.912     0.971
sobel_mae           0.02262   0.01657   0.00956
```

Per-skin at 10k:

```text
DarkSide  mae=0.00672  hit5=0.949
BlueCurve mae=0.00981  hit5=0.882
Zelda     mae=0.01213  hit5=0.905
```

10k aggregate met the MAE acceptance bound but per-file hit5 was still soft
on balance, eqmain, and main. 20k brought every per-skin hit5 above 0.95
without overfit signs. No basis for early-stop at 10k for Gate B.

## Comparison

### vs. V5 Gate A (BlueCurve, single skin)

```text
metric                       V5 Gate A (1 skin)  V5 Gate B (BlueCurve subset of 3)
exported_mae                 0.00212             0.00409
hit_5_255                    0.992               0.974
sobel_mae                    0.00296             0.00530
EQMAIN hit5                  0.947               0.879
PLEDIT hit5                  0.997               0.979
```

Sharing capacity across 3 skins costs roughly 2x MAE and ~0.02 hit5 on
BlueCurve relative to a BlueCurve-specialized V5. This is the expected and
acceptable cost of multi-skin training.

### vs. V4 Gate 2 (single-skin baselines)

```text
skin       metric       V4 single-skin    V5 Gate B (in 3-skin)
DarkSide   mae          0.00431 @10k      0.00388 @20k       lower
DarkSide   hit5         0.971   @10k      0.981   @20k       higher
DarkSide   eqmain mae   0.00879 @10k      0.00899 @20k       comparable
DarkSide   eqmain hit5  0.940   @10k      0.935   @20k       comparable

Zelda      mae          0.00421 @20k      0.00864 @20k       2x higher
Zelda      hit5         0.955   @20k      0.958   @20k       comparable
Zelda      eqmain mae   0.01461 @20k      0.01261 @20k       lower
Zelda      eqmain hit5  0.754   @20k      0.780   @20k       slightly higher
```

V5 in a 3-skin context matches or beats V4 single-skin overfits on DarkSide
and ties V4 on Zelda for the per-skin hit5 / EQMAIN metrics that V4 Gate 2
flagged. Zelda aggregate MAE is 2x higher than V4 single-skin, but this is
within the multi-skin budget shown by the BlueCurve A→B comparison above.

The EQMAIN-on-saturated-skins caveat from V4 Gate 2 persists: Zelda EQMAIN
hit5 0.78 is the same structural issue (gold panel pixels close-in-mean but
outside 5/255 tolerance), unchanged by V5 architecture.

## Export

`.wsz` exported from `snapshot_step020000` for one view of each skin:

```text
runs/slotnet_v5_3skin_gateB/export_skin_180ffb08/skin.wsz   BlueCurve
runs/slotnet_v5_3skin_gateB/export_skin_127876f0/skin.wsz   DarkSide
runs/slotnet_v5_3skin_gateB/export_skin_3cc38af4/skin.wsz   Zelda
```

All three:

```text
unzip -t                 No errors detected
file count               16
PLEDIT.bmp present       yes
VIDEO.bmp absent         yes
total archive bytes      823212
```

No replay params, distortion side-channel, prior-atlas, full-padded-atlas, or
target-view-preview artifacts are produced. Export surface matches the
restricted Cranamp contract.

## Artifacts

```text
runs/slotnet_v5_3skin_gateB/snapshot_step005000.safetensors
runs/slotnet_v5_3skin_gateB/snapshot_step010000.safetensors
runs/slotnet_v5_3skin_gateB/snapshot_step020000.safetensors
runs/slotnet_v5_3skin_gateB/last.safetensors
runs/slotnet_v5_3skin_gateB/best.safetensors
runs/slotnet_v5_3skin_gateB/eval_snapshot_step005000.json
runs/slotnet_v5_3skin_gateB/eval_snapshot_step010000.json
runs/slotnet_v5_3skin_gateB/eval_snapshot_step020000.json
runs/slotnet_v5_3skin_gateB/retrieval_snapshot_step020000.json
runs/slotnet_v5_3skin_gateB/export_skin_180ffb08/
runs/slotnet_v5_3skin_gateB/export_skin_127876f0/
runs/slotnet_v5_3skin_gateB/export_skin_3cc38af4/
data_v5_gateB_3skin/train.csv
```

## Next

Gate B passes. Proceed to Gate C: repeat `data_v4_16skin/train.csv` from
scratch against the V4 baseline, and ask whether V5 closes the V4 Gate 3
reconstruction gap (V4 Gate 3 had identity separation pass but reconstruction
fail).

Suggested Gate C plan:

```text
1. Train from scratch with --model-version 50 on data_v4_16skin/train.csv.
2. Step count likely 30k-50k (16 skins, more capacity demand).
3. Acceptance: retrieval top1 = 1.0, aggregate exported MAE < 0.02,
   per-skin hit5 > 0.85, .wsz export clean for at least 4 representative
   skins.
4. Compare against runs/slotnet_v4_16skin_masked_mem (V4 Gate 3 baseline).
```

Open caveats to watch in Gate C:

- Saturated-EQMAIN hit5 (Zelda-style): may be worse on 16 skins; track per-skin.
- BALANCE on saturated/high-gamut skins: same pattern as Zelda BALANCE here.
- Whether the per-file cross-attention helps reconstruction on 16 skins, or
  whether identity-only separation reappears as it did on V4.
