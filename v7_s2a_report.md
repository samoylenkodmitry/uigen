# V7.1 Gate S2a — cross-skin state-expansion generalization (FAIL)

Run: `uigen-kaggle-s2-state-gated`, gated StateFamilyExpander, **no skin
embedding** (deployable conditioning: source frame + source_idx + target_idx +
family_id + file_id → target frame). 14 skins, **10 train / 4 held out**
(the_four_horsemen, blair_razor, tvxq, infected_fx). gate_loss 0.05,
difficulty-weighted families, mixed local/global pair sampler, 150k steps.
Region-split eval (changed = |src-tgt|.amax>5/255 & support).

## Result

| split | changed_hit5 mean / median | unchanged_min | gate_gap |
|---|---|---|---|
| train (seen skins, seen pairs) | 0.934 / 0.982 | 0.992 | 0.998 |
| seen-skin / unseen-pair (sliders only) | 0.737 / 0.753 | 0.998 | 0.988 |
| **held-out skin (unseen style)** | **0.062 / 0.001** | 0.664 | **0.147** |

Held-out trajectory is flat across the whole run: changed_hit5 0.014 (10k) →
0.062 (150k). Per-family on held-out: 15/16 families ≈ 0.00 changed_hit5; the
gate barely opens (gap ~0.0–0.2, several negative), and unchanged pixels are
degraded (0.66–0.96 vs ~1.0 on train) — the gate misfires on unseen styles. The
only exception, BALANCE/slider 0.93, is noise (1 of 2 non-degenerate held-out
skins for that family).

By file type, held-out: buttons 0.002, sliders 0.311, toggles 0.006.

## Interpretation

- The gated expander **learns transitions within seen styles** (train 0.93) and
  **generalizes across unseen frame *pairs*** on seen skins (seen-pair sliders
  0.74, still rising — undertrained, not ceilinged).
- It does **not** transfer to an unseen **skin style** from the source frame
  alone. The transition rule is entangled with the train styles; given a
  never-seen style, the model can't even locate the change.
- This is the pre-registered "seen-skin passes, held-out fails → needs a
  context/style encoder" outcome. The source frame is present but is not being
  used as a usable style code.

## What this does and doesn't say

- Confirms (with S1) that within-style state expansion is real and learnable.
- The bottleneck for deployability is **cross-style generalization**, not the
  gated mechanism, the metric, or the task definition.
- It does NOT implicate panel/texture completion or the observer path — those
  remain separate, untouched subproblems.

## Open fork for the architecture decision (for GPT-5.5 Pro)

1. **Style/context encoder.** Derive a style embedding from the source frame
   (and/or other observed assets of the same skin) and condition the
   transition + gate on it. This is the deployable analogue of the oracle
   skin-id. Likely the main fix.
2. **Augmentation.** Heavy source-frame color/material augmentation during
   training to force a style-agnostic, position/structure-based change rule
   (the gate should localize the change by geometry, not learned style).
3. **More train styles.** Only 10 train skins; cross-style generalization may
   be data-starved. Cheap to test (shift the train/heldout split), but unlikely
   to fully close a 0.06 vs 0.93 gap.
4. **S2b oracle upper bound (diagnostic).** Re-run with oracle skin embedding on
   the seen-skin splits to confirm "transition is learnable given style context"
   — though S1 (single skin) already strongly implies yes. Oracle cannot help
   genuinely held-out skins (untrained ids), so it is a diagnostic, not a fix.

Recommendation: the real next step is (1), a style/context encoder, optionally
with (2). That is an architecture change worth a deliberate design pass, not
another ad-hoc training loop.
