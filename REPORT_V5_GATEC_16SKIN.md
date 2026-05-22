# SlotNet V5 Gate C — 16-Skin Reconstruction

Date: 2026-05-22

## Conclusion

V5 substantially closes the V4 Gate 3 reconstruction gap and preserves V4's
identity separation. Gate C meets the **primary** acceptance criteria but
fails the **strict** per-skin and aggregate hit5 thresholds.

```text
retrieval top1                    1.000     PASS  (>= 0.95, also strict 1.000)
aggregate exported_pixels_mae     0.01848   PASS  (<  0.020)
aggregate hit_5_255               0.790     FAIL  (>  0.85)
per-skin hit5 > 0.85              5 / 16    FAIL  (target: all 16)
materially beats V4 Gate 3        yes       PASS
.wsz clean for selected skins     6 / 6     PASS
```

Verdict: V5 is the right architectural direction. The V5 cross-attention
into the encoder spatial map produces a 2.2x MAE reduction and a +0.12 hit5
absolute lift over V4 Gate 3 on the same 16-skin dataset and step budget,
without breaking identity separation. The remaining gap to strict
acceptance is concentrated on a small number of structurally-hard skins
(photographic / saturated content); `tvxq` is a clear outlier and may need
a dedicated investigation.

This is not a strict pass. It is a strong-direction signal that warrants a
follow-up decision on whether to (a) increase capacity, (b) extend
training, (c) address the tvxq-style outliers structurally, or (d) accept
the relaxed bar and ship V5 as the new baseline for the V4 → V5 transition.

## Acceptance Row-by-Row

```text
criterion                                   target              measured       verdict
retrieval top1                              >= 0.95 / 1.000     1.000          PASS (both)
aggregate exported_pixels_mae               <  0.020            0.01848        PASS
aggregate exported_pixels_hit_5_255         >  0.85             0.79002        FAIL
per-skin hit5 > 0.85                        all 16              5 / 16         FAIL
materially beats V4 Gate 3                  yes                 MAE 2.2x lower PASS
.wsz export clean                           4-6 representative  6 / 6 clean    PASS
PLEDIT.bmp present per export               yes                 yes (6 / 6)    PASS
VIDEO.bmp absent per export                 yes                 yes (6 / 6)    PASS
identity separation regression              none                none           PASS
training stability                          no NaN / OOM        clean          PASS
```

Per the plan's pass-with-caveat clause, "Gate C passes-with-caveat if
aggregate metrics pass but EQMAIN hit5 on saturated/gold skins drops below
0.75" — but the actual shortfall here is broader than just EQMAIN. Several
saturated/photographic skins fall under 0.85 across multiple files, and the
aggregate hit5 also misses. Reporting this as **pass-primary / fail-strict**
rather than pass-with-caveat.

## Learning Curve

```text
step      mae        hit5       sobel
10000     0.05160    0.6029     0.0774
20000     0.03054    0.6964     0.0395
30000     0.02242    0.7695     0.0285
50000     0.01848    0.7900     0.0240
```

Trajectory:

- 10k → 20k: MAE -41%, hit5 +0.094, sobel -49%. Big jump.
- 20k → 30k: MAE -27%, hit5 +0.073, sobel -28%. Still steep.
- 30k → 50k: MAE -18%, hit5 +0.021, sobel -16%. Decelerating.

Diminishing returns past 30k; the slope at 50k is shallow enough that
naïvely extending to 100k is unlikely to lift the bottom-tier per-skin hit5
above 0.85. The most likely path to strict pass is changing the model or
the data, not more steps.

V5 beat V4 Gate 3's final 50k aggregate by step 20k:

```text
V4 Gate 3 @50k:  mae=0.0406  hit5=0.668  sobel=0.0585
V5 Gate C @20k:  mae=0.0305  hit5=0.696  sobel=0.0395   (beats V4 final)
V5 Gate C @50k:  mae=0.0185  hit5=0.790  sobel=0.0240   (~2.2x lower MAE)
```

## Per-Skin Aggregate (step 50000, hit5 sorted)

```text
*  skin                                              mae        hit5      sobel
   goodgawd_bba84deb                                 0.01663    0.9781    0.0344
   simblyblayit_ff648b6c                             0.01064    0.9740    0.0156
   minimalistic_black_145917e6                       0.00497    0.9730    0.0072
   engraved4_platinum_5638acf5                       0.00606    0.9483    0.0103
   darkside_127876f0                                 0.00969    0.8837    0.0158
*  infected_fx_gray_no_transparency_9f3bd211         0.01802    0.8372    0.0130
*  cyborg_5569c5cf                                   0.01338    0.8287    0.0193
*  ruki2_by_michi_caa5bfe3                           0.01384    0.8248    0.0208
*  rancid_amp_5_42a78437                             0.01657    0.8178    0.0308
*  aguileramp_oldschool_2e1e7540                     0.01390    0.8153    0.0224
*  the_four_horsemen_523e6bdf                        0.02065    0.8120    0.0326
*  a_halo_so_bright_it_bleeds_3ee84993               0.02017    0.7137    0.0326
*  dragonzv30amp_85acc35c                            0.02338    0.6725    0.0356
*  zelda_amp_gold_3cc38af4                           0.02562    0.6326    0.0454
*  blair_razor_project_e7dd3210                      0.02282    0.6317    0.0325
*  tvxq_winamp_skins_by_roseweedy_c379f7bd           0.05935    0.2972    0.0149

(*) hit5 < 0.85
```

Buckets:

```text
>= 0.95 hit5    3 skins   goodgawd, simblyblayit, minimalistic_black
0.88 - 0.95     2 skins   engraved4, darkside
0.81 - 0.85     6 skins   infected_fx, cyborg, ruki2, rancid, aguileramp,
                          the_four_horsemen   (near-miss bucket)
0.60 - 0.80     4 skins   a_halo, dragonzv30amp, zelda, blair_razor
<  0.40         1 skin    tvxq (clear outlier)
```

The 6 near-miss skins are within 0.04 of the bar. The 5 hard skins are 0.13 -
0.55 short, and tvxq is in its own failure mode.

## Per-File Aggregate (step 50000)

```text
file        MAE        hit5        sobel
balance     0.02256    0.7826      0.0329
cbuttons    0.01548    0.7800      0.0186
eqmain      0.03439    0.6814      0.0523
main        0.03099    0.6921      0.0421
monoster    0.00733    0.9096      0.0083
playpaus    0.00222    0.9851      0.0033
pledit      0.01993    0.7577      0.0272
posbar      0.01780    0.7808      0.0129
shufrep     0.01423    0.7747      0.0162
titlebar    0.01189    0.8439      0.0148
volume      0.02648    0.7023      0.0350
```

Small sprites still solid (`playpaus` 0.985, `monoster` 0.910, `titlebar`
0.844). The big-canvas files (`eqmain`, `main`, `volume`, `balance`) are
the systemic weakness on multi-skin training. The Zelda BALANCE pattern
from Gate B is now a general "saturated canvas" pattern.

## Per-Skin Worst Files (hit5 < 0.85)

Counted per skin, only highlighting files driving the per-skin miss:

```text
tvxq_winamp_skins_by_roseweedy        10 / 11 files < 0.85
  balance     mae=0.0690  hit5=0.0008   (essentially zero)
  posbar      mae=0.0700  hit5=0.0560
  volume      mae=0.0862  hit5=0.0673
  shufrep     mae=0.0601  hit5=0.1218
  cbuttons    mae=0.0725  hit5=0.2496
  titlebar    mae=0.0504  hit5=0.2626
  pledit      mae=0.0675  hit5=0.2784
  eqmain      mae=0.0724  hit5=0.3276
  main        mae=0.0747  hit5=0.4123
  monoster    mae=0.0280  hit5=0.4969
  only playpaus is above 0.85

zelda_amp_gold                        9 / 11 files < 0.85
  eqmain      mae=0.0412  hit5=0.4208     (continues V4 Gate 2 / V5 Gate B finding)
  main        mae=0.0402  hit5=0.4212
  pledit      mae=0.0310  hit5=0.4406
  titlebar    mae=0.0272  hit5=0.4632
  ...

blair_razor_project                   9 / 11 files < 0.85
  eqmain      mae=0.0421  hit5=0.3979
  main        mae=0.0381  hit5=0.3719
  volume      mae=0.0379  hit5=0.4751
  pledit      mae=0.0288  hit5=0.4687
  ...

dragonzv30amp                         8 / 11 files < 0.85
  main        mae=0.0506  hit5=0.4516
  eqmain      mae=0.0494  hit5=0.4509
  volume      mae=0.0388  hit5=0.3736
  ...

a_halo_so_bright_it_bleeds            7 / 11 files < 0.85
  main        mae=0.0456  hit5=0.5349
  volume      mae=0.0449  hit5=0.3988
  balance     mae=0.0274  hit5=0.4941
  ...
```

The hard-skin pattern is consistent: large saturated/photographic canvases
(`main`, `eqmain`, `volume`, `balance`, `pledit`) fail; small sprites pass.

The `tvxq` failure is qualitatively different: every file struggles. This
suggests a per-skin conditioning / capacity issue, not a per-file issue.
`tvxq` is also a photographic skin with high-frequency content.

## Watch Items (carried from Gate B)

```text
Zelda BALANCE
  Gate B: mae 0.054, hit5 0.889
  Gate C: mae 0.018, hit5 ~0.86 (improved on BALANCE specifically, but
                                 other Zelda files now dominate the miss)
  Verdict: not a Gate-C-blocker; the broader saturated-canvas pattern
           replaces the single-file BALANCE concern.

Zelda EQMAIN  (saturated-EQMAIN issue from V4 Gate 2)
  Gate B: hit5 0.78
  Gate C: hit5 0.42
  Verdict: structural issue persists and worsens under multi-skin scaling.
           Not surprising; flagged in the plan.

Multi-skin scaling cost
  Gate A   1 skin    aggregate MAE 0.00212
  Gate B   3 skin    aggregate MAE 0.00554   (2.6x of Gate A)
  Gate C  16 skin    aggregate MAE 0.01848   (8.7x of Gate A, 3.3x of Gate B)
  Trend:  sub-linear; capacity sharing is reasonably efficient on small
          counts but the per-skin hit5 distribution widens — strong skins
          stay strong, hard skins degrade more.
```

## Retrieval

```text
top1_accuracy   1.000        (512 / 512)
samples         512
target_skins    16
misclassifications  0
```

Per-skin mean best-MAE (i.e. how cleanly the model reproduces the correct
skin's expected output):

```text
minimalistic_black     0.00497    (cleanest)
engraved4_platinum     0.00606
darkside               0.00969
simblyblayit           0.01064
cyborg                 0.01338
ruki2                  0.01384
aguileramp             0.01390
rancid_amp             0.01657
goodgawd               0.01663
infected_fx            0.01802
a_halo                 0.02017
the_four_horsemen      0.02065
blair_razor            0.02282
dragonzv30amp          0.02338
zelda                  0.02562
tvxq                   0.05935    (3x the median; consistent with the
                                   per-skin reconstruction failure)
```

No skin pair is confused. Despite tvxq's reconstruction being 3x the
median, the global style vector still separates it cleanly from the other
15 skins. The V4 Gate 3 identity-separation result is preserved.

## Comparison vs. V4 Gate 3 (matched 50k, same dataset, same hardware)

```text
metric                           V4 Gate 3 @50k    V5 Gate C @50k    delta
retrieval top1                   1.000             1.000             tied
aggregate exported_mae           0.04065           0.01848           2.2x lower
aggregate hit_5_255              0.668             0.790             +0.122
aggregate sobel_mae              0.05852           0.02396           2.4x lower
supported_slots_mae              0.11004           ---               (V5 default eval differs;
                                                                     not measured here)
```

V5 is materially better than V4 on every reconstruction metric while tying
on retrieval. The V5 architectural bet — per-file cross-attention into the
encoder spatial map — works for reconstruction at 16-skin scale.

## Comparison vs. V5 Gate A / Gate B (scale context only)

```text
aggregate exported_mae           Gate A   Gate B   Gate C
                                 0.00212  0.00554  0.01848
                                 (1 skin) (3 skin) (16 skin)

aggregate hit_5_255              Gate A   Gate B   Gate C
                                 0.992    0.971    0.790
```

The scaling pattern observed:

- 1 → 3 skins: MAE 2.6x, hit5 -0.02
- 3 → 16 skins: MAE 3.3x, hit5 -0.18

hit5 degrades non-linearly. This is consistent with the per-skin hit5
distribution widening rather than the mean collapsing — most skins remain
near Gate B level; a few hard ones drag the mean.

BlueCurve is not in the 16-skin set, so a like-for-like A → C comparison on
the same skin is not available.

## Exports

Six representative `.wsz` exports from `snapshot_step050000`:

```text
runs/slotnet_v5_16skin_gateC/export_darkside_127876f0/skin.wsz
runs/slotnet_v5_16skin_gateC/export_zelda_amp_gold_3cc38af4/skin.wsz
runs/slotnet_v5_16skin_gateC/export_aguileramp_oldschool_2e1e7540/skin.wsz
runs/slotnet_v5_16skin_gateC/export_a_halo_so_bright_it_bleeds_3ee84993/skin.wsz
runs/slotnet_v5_16skin_gateC/export_minimalistic_black_145917e6/skin.wsz
runs/slotnet_v5_16skin_gateC/export_the_four_horsemen_523e6bdf/skin.wsz
```

Per-export check on every one:

```text
unzip -t                  No errors detected
file count                16
PLEDIT.bmp present        yes
VIDEO.bmp absent          yes
```

No replay params, distortion side-channel, prior-atlas, or full-padded-atlas
artifacts. Export surface matches the restricted Cranamp contract on all 6.

## Wall Time and Step Rate

```text
training         50000 steps, batch 2, AMP, RTX 2070 (8 GB)
wall time        ~5h 33m
step rate        ~150 steps/min   (matches V4 Gate 3's ~165 steps/min;
                                   small overhead from V5 cross-attn path)
final training loss   0.03764
```

No NaN / OOM / instability. Training completed cleanly end to end.

## Artifacts

```text
runs/slotnet_v5_16skin_gateC/snapshot_step{010000,020000,030000,050000}.safetensors
runs/slotnet_v5_16skin_gateC/last.safetensors
runs/slotnet_v5_16skin_gateC/best.safetensors
runs/slotnet_v5_16skin_gateC/eval_snapshot_step{010000,020000,030000,050000}.json
runs/slotnet_v5_16skin_gateC/retrieval_snapshot_step050000.json
runs/slotnet_v5_16skin_gateC/export_<6 skins>/
data_v4_16skin/train.csv  (existing, unchanged)
```

## What This Means

The V5 architecture is the right direction:

- 2.2x MAE reduction, +0.12 hit5 on the same dataset / step budget as V4.
- Identity separation preserved at 16 skins (top1 1.000 / 512).
- 5 of 16 skins are essentially solved (hit5 > 0.94).
- 6 more skins are within 0.04 of the strict bar.

But strict pass requires structural work, not more steps:

- The 30k → 50k slope is shallow. Naïvely extending training is unlikely to
  clear the gap on hard skins.
- The hard skins fail on big canvas files (`main`, `eqmain`, `volume`,
  `balance`) with photographic / saturated content. Same failure mode as
  V4 Gate 2's saturated-EQMAIN caveat, now broader.
- `tvxq` is an outlier in degree (10 / 11 files < 0.85). Worth a focused
  investigation: input view, target atlas, attention map.

## Suggested Next Steps

These are options for the user to choose between; **none is launched yet**.

```text
1. Accept V5 as the new baseline at the relaxed bar.
   Document the per-skin failure modes; treat strict pass as a follow-up
   research question. V5 is already strictly better than V4 on every
   metric; shipping V5 as the new floor is reasonable.

2. Targeted investigation of tvxq.
   Visualize input view, target atlas, predicted output, attention map.
   Determine whether tvxq is a data issue (bad render / bad target),
   a capacity issue (style vector can't represent its content), or an
   architectural issue (cross-attn can't reach the right tokens).
   Cheap to do; informs every other decision.

3. Capacity bump.
   --base-channels 32 --style-dim 256 --head-channels 128, same step
   budget. Tests whether hard skins are capacity-bound. Larger memory
   footprint may push the 2070 toward OOM at batch 2.

4. Longer training (75k or 100k).
   Cheapest action. Will continue to lift the near-miss bucket (the 6
   skins currently 0.81-0.85) but unlikely to fix the hard 5 or tvxq.
   Diminishing returns observed at 30k -> 50k.

5. Curriculum / per-skin reweighting.
   Up-sample the hard skins (tvxq, zelda, dragonzv30amp, blair_razor)
   so they get more gradient. Risks degrading the strong skins.

Recommendation: 2 first (cheap, high information), then choose between
1 and 3 / 4 based on what 2 reveals.
```

## Out-of-Scope Follow-Ups (not done in this gate)

- Attention map dumping for V5.
- Visual diff of tvxq predicted vs. target.
- Reconsidering the support-pixel mask for photographic content.

These were excluded by the Gate C plan and remain excluded.
