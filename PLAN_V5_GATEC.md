# Gate C Plan — V5 on 16 skins

Status: **draft, awaiting review**. Do not launch until approved.

## Question

Does V5's per-file cross-attention into the encoder spatial map close the
V4 Gate 3 reconstruction gap, while preserving the identity separation
V4 Gate 3 already achieved?

V4 Gate 3 @50k: retrieval `1.000`, but exported MAE `0.04065`, hit5 `0.668`.
Identities separated cleanly; per-file decoders underfit details.

## Training

Dataset (existing):

```text
data_v4_16skin/train.csv          512 rows (16 skins x 32 distorted views)
```

From scratch (no resume from Gate A/B):

```bash
.venv/bin/python train_slotnet.py \
  --model-version 50 \
  --train data_v4_16skin/train.csv \
  --steps 50000 \
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
  --checkpoint-every 2000 \
  --snapshot-every 2000 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --amp \
  --out runs/slotnet_v5_16skin_gateC \
  --device cuda
```

Notes vs. Gate A/B:

- `--steps 50000` matches V4 Gate 3 for apples-to-apples comparison.
- `--checkpoint-every 2000` / `--snapshot-every 2000` to keep snapshot count
  manageable (25 snapshots vs. 50 at 1k cadence; each snapshot ~23 MB).
- All other hyperparameters identical to Gate B. Do not vary architecture
  or losses inside Gate C — the question is purely about scale.

Expected wall time: ~5–5.5h on RTX 2070 (V4 Gate 3 ran ~5h at the same
batch/AMP settings on the same hardware).

## Evaluation Cadence

Intermediate checkpoints to eval:

```text
step 10000   first real check; if much worse than V4 at the same step, pause
step 20000   should comfortably beat Gate B aggregate; expected hit5 > 0.85
step 30000   Gate B-level fit on individual skins
step 50000   final / acceptance
```

Each intermediate eval blocks until GPU is free *or* runs on CPU.
At 512 rows × 11 files, CPU eval will be ~5–10 min. Acceptable.

Decision rule:

- Step 10k is a smoke / trajectory check, not a pass/fail checkpoint. Compare
  against V4 only if a matched V4 10k full-dataset eval is already available;
  otherwise compare against the final V4 Gate 3 baseline and the current V5
  learning slope.
- Otherwise continue to 50k. Do not early-stop unless 30k already beats
  V4 Gate 3 by 3x+ on MAE *and* per-skin hit5 > 0.90 everywhere.

## Acceptance Thresholds

Primary Gate C target:

```text
retrieval top1                    >= 0.95      (1.000 is ideal)
aggregate exported_pixels_mae     < 0.020      (V4 Gate 3 = 0.04065; halve it)
aggregate hit_5_255               > 0.85       (V4 Gate 3 = 0.668)
per-skin hit5                     > 0.85       for all 16 skins
.wsz export                       clean        for 4-6 representative skins
PLEDIT.bmp / VIDEO.bmp policy     respected    on every export
```

Strict success:

```text
retrieval top1                    1.000
aggregate exported_pixels_mae     < 0.020
aggregate hit_5_255               > 0.85
per-skin hit5                     > 0.85 for all 16 skins
```

Gate C passes-with-caveat if aggregate metrics pass but EQMAIN hit5 on
saturated/gold skins (Zelda-family) drops below 0.75. Note as a structural
follow-up; do not block Gate C on it.

Hard fail if retrieval top1 drops below `0.95` or if the reconstruction metrics
do not materially beat V4 Gate 3. If retrieval is between `0.95` and `1.000`,
report it as identity-separation regression even if reconstruction improves.

## Reporting

`REPORT_V5_GATEC_16SKIN.md` to include:

```text
1. Acceptance row-by-row (each threshold + measured + verdict).
2. Aggregate metrics at 10k / 20k / 30k / 50k (learning curve).
3. Per-skin aggregate at 50k (16 rows; mae / hit5 / sobel).
4. Per-file aggregate at 50k (11 rows; mae / hit5 / sobel).
5. Per-skin outlier table:
   - Top-3 worst MAE files per skin (by mae).
   - Top-3 worst hit5 files per skin.
   - Explicitly track per-skin BALANCE and EQMAIN.
6. Retrieval table:
   - top1, top5, confusion entries for any miss.
   - per-skin best-MAE distribution.
7. Comparison vs. V4 Gate 3 (matched table on every aggregate metric).
8. Comparison vs. V5 Gate B (3 skin) and V5 Gate A (BlueCurve, not in the
   16-skin set) as scale-context only. BlueCurve is not part of Gate C.
9. Wall time and step rate.
10. 4-6 .wsz exports listed with integrity, PLEDIT/VIDEO check.
```

## Watch Items (from Gate B)

Carry these forward and report explicitly even if aggregate passes:

```text
- Zelda BALANCE: Gate B mae 0.054, hit5 0.889. May worsen on 16-skin or may
  reveal a per-skin BALANCE pattern (saturated/high-gamut skins).
- Zelda EQMAIN hit5 0.78: recurring saturated-EQMAIN issue from V4 Gate 2.
  Likely to persist; not a Gate C blocker.
- Multi-skin scaling cost: Gate A -> Gate B roughly doubled BlueCurve MAE, but
  BlueCurve is not in the 16-skin Gate C dataset. Use Gate B only as scale
  context; Gate C's bar is the matched V4 Gate 3 comparison.
```

## Exports

Export these six representative skins:

```text
continuity / known issues:
  darkside_127876f0
  zelda_amp_gold_3cc38af4
  aguileramp_oldschool_2e1e7540

novel coverage:
  a_halo_so_bright_it_bleeds_3ee84993
  minimalistic_black_145917e6
  the_four_horsemen_523e6bdf
```

Rationale:

```text
darkside / zelda / aguileramp   continuity with previous gates
a_halo                          bright/saturated photographic
minimalistic_black              dark low-palette flat UI
the_four_horsemen               high-detail / high edge-density texture
```

BlueCurve is not in `data_v4_16skin/train.csv`; do not export it for Gate C.

## Out of Scope

```text
- Architecture changes.
- Loss / mask changes.
- Resume from Gate A or Gate B checkpoints.
- Visual attention-map dumping (carry as a follow-up; not Gate C blocker).
- The Zelda BALANCE root-cause deep dive (do a quick diff later if Gate C
  surfaces a BALANCE-wide pattern; otherwise defer).
- Touching V4 code.
```

## Risks and How They Surface

```text
Risk: per-file cross-attention does not scale to 16 skins.
  Surface: step 10k MAE no better than V4 Gate 3 step 10k.
  Action: pause, inspect attention maps and per-file decoder gradients.

Risk: identity confusion appears at 16 skins.
  Surface: retrieval top1 < 1.000 at any eval step.
  Action: pause, investigate which skin pairs are confused; this would
  contradict V4 Gate 3's clean identity result and warrant deeper review.

Risk: training instability (NaN / runaway loss).
  Surface: training crashes or NaN appears in metrics.
  Action: usual debugging; check for AMP overflow path.

Risk: wall time blows out.
  Surface: >7h with no convergence sign.
  Action: stop, inspect, do not throw more steps at it without a hypothesis.
```

## Approval

Before launching:

- Confirm step count: `50000` (matches V4 Gate 3).
- Confirm snapshot cadence: every `2000` steps.
- Confirm acceptance thresholds above.
- Export the six representative skins listed above.
- Confirm CPU-eval cadence (10k / 20k / 30k / 50k) is acceptable, or relax
  to (20k / 50k only) if intermediate CPU evals are too slow.
