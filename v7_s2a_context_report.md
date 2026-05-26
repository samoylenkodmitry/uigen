# V7.1 Gate S2a-context — source style/context encoder (FAIL, no lift-off)

GPT-5.5 Pro's fix for the S2a cross-skin failure was a deployable style/context
encoder: a global code derived from `source_rgb` only (no oracle skin id),
broadcast into the transition/gate path (`--style-context-dim 64`,
model_version 73). This is the early-read run for that fix.

Run: `uigen-kaggle-s2-context`. **Byte-identical to the failed S2a run except**
`--style-context-dim 64` and `STEPS=50000` (early read; failed S2a was flat the
whole 150k, so lift-off is decidable well before 150k). Same 14 skins, same 4
held out, same sampler / gate loss 0.05 / eval snapshots, NO oracle skin emb.

## Result: FLAT — the style code did not move held-out generalization

Held-out skin (the deployability metric), region-split eval:

| step | held changed_hit5 mean / median | held min-family | held unchanged_min | gate_gap | seen-pair changed |
|---|---|---|---|---|---|
| 10k | 0.013 / — | 0.000 | 0.359 | 0.143 | 0.033 |
| 20k | 0.033 / — | 0.000 | 0.556 | 0.115 | 0.267 |
| 30k | 0.037 / — | 0.000 | 0.552 | 0.091 | 0.308 |
| 40k | 0.052 / — | 0.000 | 0.544 | 0.126 | 0.451 |
| **50k** | **0.054 / 0.001** | **0.000** | **0.519** | **0.106** | 0.462 |

Side-by-side with the failed no-context S2a:

| | held changed mean | held median | gate_gap | families <0.85 | shape |
|---|---|---|---|---|---|
| **no-context S2a** @150k | 0.062 | 0.001 | 0.147 | 15/16 ≈ 0 | flat 0.014→0.062 |
| **context-64 S2a** @50k | 0.054 | 0.001 | 0.106 | 16/16 ≈ 0 | flat 0.013→0.054, plateauing |

Held-out per-family @50k: only **BALANCE/slider 0.75** (the same known noise
artifact — few non-degenerate held-out skins for it) and **PLAYPAUS/status
0.10** are non-dead; the other 14 families are ~0.00. The held-out curve is
*plateauing* (increments shrinking: +0.020, +0.004, +0.015, +0.002), not
accelerating — extending to 150k will not change the verdict. **Did not extend.**

Seen-skin learning is healthy (seen-pair rising 0.03→0.46 by 50k; train loss
~0.005), so the context path is not breaking seen-skin training — it simply does
not transfer.

## Interpretation

The global, average-pooled style code from a single source frame does **not**
enable cross-skin transfer. The failure shape is unchanged from no-context: on
an unseen style the gate cannot even **locate** the change region (gate_gap
~0.1), so changed pixels are never written.

Two things this strongly suggests:

1. **The style code behaves as a soft per-skin id, not a generalizable style
   rule.** Seen skins fine, held-out dead is the classic soft-oracle pattern:
   the encoder maps each train skin's source-frame statistics to that skin's
   transition behavior; an unseen skin's code lands out-of-distribution and the
   conditioned transition collapses.
2. **A globally-pooled vector discards spatial structure.** `AdaptiveAvgPool2d(1)`
   conveys palette/material but not *where the button is* on an unseen skin. The
   gate's job is fundamentally spatial (localize the changed region), and a
   global code gives it no spatial cue. The raw source RGB is in the input but
   the model is not learning to use source structure to localize changes in a
   style-invariant way.

## Open fork for GPT-5.5 Pro (architecture)

- **(A) Spatial/structural conditioning for the gate, not a global vector.**
  The missing signal is *where to look* on an unseen skin. Note: in classic
  Winamp skins the per-family sprite geometry within each BMP is fixed across
  skins (that's what makes them classic-compatible) — `state_families_classic.yaml`
  already encodes those rects. Feeding the family's region as a spatial
  conditioning channel/prior could tell the gate where the alternative lives
  regardless of style. This is an unused, skin-invariant signal.
- **(B) Multi-asset context bank** (codex's pre-registered fallback): derive the
  style/context code from *several* observed same-skin assets, not one source
  crop, so the code carries more than single-frame statistics.
- **(C) Heavy source augmentation** (palette/material jitter) to force a
  geometry-based, style-agnostic change rule and break the soft-id memorization.
- **(D) FiLM-style injection** instead of concat (modulate U-Net features by the
  code at multiple depths) — cheaper to test, but unlikely to fix the spatial
  blindness in (A).

My read: (A) is the highest-leverage next step — the gate is being asked to
relearn each family's location per skin from scratch, and we already know that
geometry. (C) likely complements it. (B)/(D) are secondary.

Code: style encoder at `models/v7_state_expander.py` (`_SourceStyleEncoder`),
committed `0d618ed`. Full suite 302 passed. No regression to seen-skin behavior.
