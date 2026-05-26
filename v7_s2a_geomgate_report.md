# V7.1 Gate S2a-geomgate — geometry gate prior (FAIL, but diagnostic)

GPT-5.5 Pro's spatial fork: an additive, skin-INDEPENDENT geometry gate prior.
`gate_logits = content_gate_logits + geometry_gate_logits`, where the geometry
branch is a per-pixel coordinate MLP over fixed classic sprite geometry
(src/tgt rects, frame indices, file size) + family id — no RGB, no skin id.
RGB/output path unchanged; `--style-context-dim 0`, `--skin-embedding-dim 0`.

Run: `uigen-kaggle-s2-geomgate`, geo_gate_hidden 64, 50k early read, same 14
skins / 4 held out / sampler / gate-loss 0.05 / eval-every-10k as the failed
S2a runs.

## Result: FLAT held-out — but the gate DID open on train

| split | changed_hit5 mean / median | unchanged_min | gate_gap |
|---|---|---|---|
| train (seen skins, seen pairs) | 0.723 / 0.829 | 0.991 | **0.995** |
| seen-skin / unseen-pair | 0.458 / 0.500 | 0.995 | 0.980 |
| **held-out skin (unseen style)** | **0.043 / 0.002** | 0.655 | **0.154** |

Held-out trajectory: changed_hit5 0.009 (10k) → 0.043 (50k), flat. Comparison
to the prior cross-skin attempts (all flat, same band):

| run | held changed_hit5 | held gate_gap |
|---|---|---|
| S2a (no context) @150k | 0.062 | 0.147 |
| S2a-context (style) @50k | 0.054 | 0.106 |
| **S2a-geomgate @50k** | **0.043** | **0.154** |

## Diagnosis: the additive content path steals the gradient

This run isolates *why* cross-skin localization fails, and it is NOT "geometry
can't localize":

- On **train** the gate opens hard (gap **0.995**, gate_on_changed ~0.99,
  on_unchanged ~0.00 for every family).
- On **held-out** the gate collapses to gap ~0.15 — **even though the geometry
  inputs are byte-identical across skins** (same families, same rects). If the
  geometry branch were doing the opening, held-out would match train.
- Therefore the train-time opening comes from the **content (RGB) gate**, not
  the geometry prior. With an additive gate and a copy-biased content gate that
  *can* open from RGB on seen skins, the content path greedily explains all the
  change during training, so the geometry branch receives almost no gradient and
  learns only a weak prior (the residual held-out gap of 0.15). The gate
  supervision (BCE on the *combined* logits) is likewise satisfied by the
  content path alone, so it doesn't force geometry to learn either.

Corroboration: the only held-out families where the geometry gate *did* open are
**BALANCE/slider gap 0.889 (changed_hit5 0.628)** and **PLAYPAUS/status gap
0.536** — i.e. where geometry bound, it produced the only non-trivial held-out
signal. Everywhere else gap < 0.22 (several negative: CBUTTONS play/pause).

So: geometry was never *forced* to be the localizer; the content path made it
redundant on the training split, and the content path is exactly what is OOD on
an unseen skin.

## What this does and doesn't say

- Confirms (again) within-style learning is fine (train 0.72, seen-pair 0.46).
- The blocker is still held-out gate localization, and we now know the additive
  design does not fix it because the RGB content gate out-competes the geometry
  prior for the gradient.
- Where the geometry gate *did* engage on held-out, it gave the only real signal
  — evidence the geometry path can localize if forced to carry the load.

## Open fork for GPT-5.5 Pro (force geometry to be the localizer)

The fix indicated by the diagnosis is to stop the content gate from out-competing
geometry. Concrete, cheap options (one 50k early read each):

1. **Pure geometry gate (recommended).** `gate = sigmoid(geometry_gate_logits)`
   only — remove the content/RGB gate entirely. The gate becomes RGB-free, so it
   *must* localize from geometry; the RGB head only writes values. This is the
   clean test of "can fixed geometry localize the change across skins," and it
   directly disambiguates the two cases GPT-5.5 Pro called out:
     - geometry gate opens on held-out (high gap) AND changed_hit5 rises → win;
     - gate opens but changed_hit5 stays low → "gate opens, RGB values wrong"
       (the DIFFERENT, later style/material problem) — now reached, cleanly;
     - gate still doesn't open → geometry genuinely can't localize (unlikely
       given BALANCE/PLAYPAUS above).
2. **Supervise the geometry logits directly.** Apply the changed-region BCE to
   `geometry_gate_logits` (not the combined logits) so the geometry path is
   forced to predict the change mask regardless of the content path.
3. **Both** (1)+(2).

Recommendation: run (1) — pure geometry gate, geometry-supervised — next. It is a
one-line change to the gate composition behind a flag, same kernel/split, 50k.

Code: geometry gate at `models/v7_state_expander.py` (`_GeometryGate`),
`pair_geom` in `atlas_ai/state_pairs_dataset.py`, committed `77d6f12`. Full
suite 306 passed. No regression to seen-skin behavior.
