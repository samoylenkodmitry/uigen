# V7 Phase 0 re-eval: Phase B checkpoint under corrected masks + hidden metrics

**Checkpoint:** `runs/gateB_curriculum_phaseB/best.safetensors`
(c48, file_emb 32, skin_emb 64, num_skins 14; oracle skin id ON). Trained
under the *old* support-normalized loss — this is a re-measurement of an
existing checkpoint, not a new run.

**Eval config:** `scripts/19_eval_v7_completer.py`, `mask_samples=8`, seed 0,
14-skin set (`data_v7_16skin_completion`), batch 14. Two mask mixes:

- **A — alternatives-only:** `state_family=1.0` (rr/pv/wf/pt = 0).
  Component-only files (no `alternatives` family) are skipped.
- **B — corrected mixed:** `state_family=0.7, random_rect=0.3`.
  All 11 files eligible (component-only files reconstruct under random_rect).

All metrics below are **hidden-normalized** (numerator/denominator over
`hidden = (1 - observed) * support`) unless labelled `full_`.

## Headline

The hidden metrics tell a very different story from the old support-normalized
ones, and the difference is not noise — it is the hard-copy dilution the
Phase 0 handoff predicted.

| | hidden_mae | hidden_hit5 | full_mae (old) | full_hit5 (old) | obs_passthrough |
|---|---|---|---|---|---|
| **A** state_family-only (7 files) | 0.0539 | 0.608 | 0.0228 | 0.834 | 0.0000 |
| **B** mixed sf0.7/rr0.3 (11 files) | 0.0654 | 0.570 | 0.0140 | 0.908 | 0.0000 |

`observed_passthrough_mae = 0.0` confirms the observed-pixel copy is exact, so
the hidden denominators are clean. The old `full_*` numbers (≈0.014 mae / 0.91
hit5) are diluted 3–4× by the hard-copied observed pixels: on the part of the
task the model must actually generate, it is at **mae ≈ 0.05–0.065, hit5 ≈
0.57–0.61 — nowhere near the gate (mae < 0.015, hit5 > 0.90).**

So the answer to the handoff's question ("was the Gate B failure partly caused
by support-normalized denominators?") is **yes, decisively** — but in the
direction of *the model was never close*, not *it was better than reported*.
The curriculum's "mix mae 0.028, sliders solved" framing was a measurement
artifact.

## Per-mode (Eval B) — the real surprise

| mode | hidden_mae | hidden_hit5 | hidden_sobel | n |
|---|---|---|---|---|
| state_family | 0.0484 | 0.618 | 0.184 | 528 |
| **random_rect** | **0.1983** | **0.196** | 0.554 | 704 |

We previously believed random_rect was "trivially solved (~0.001)". That was
the support-normalized illusion: random_rect hides a small rect, so most
supported pixels are observed/hard-copied and the support-normalized mae looked
tiny. **Hidden-normalized, random_rect is the model's worst mode by far** — it
essentially never learned to inpaint an arbitrary hidden rectangle. It coasted
on the copy and learned only the structured state-family frames it was drilled
on.

## Per-file hidden (Eval B unless skipped in A)

| file | A mae | A hit5 | B mae | B hit5 | B full_mae (old) |
|---|---|---|---|---|---|
| VOLUME   | 0.0166 | 0.738 | 0.0164 | 0.757 | 0.0107 |
| BALANCE  | 0.0237 | 0.576 | 0.0245 | 0.565 | 0.0168 |
| PLAYPAUS | 0.0394 | 0.707 | 0.0380 | 0.700 | 0.0181 |
| MONOSTER | 0.0592 | 0.400 | 0.0537 | 0.413 | 0.0199 |
| SHUFREP  | 0.1381 | 0.215 | 0.1454 | 0.225 | 0.0167 |
| POSBAR   | (skipped) | | 0.1578 | 0.244 | 0.0073 |
| CBUTTONS | 0.1692 | 0.150 | 0.1735 | 0.149 | 0.0118 |
| EQMAIN   | 0.2032 | 0.332 | 0.1871 | 0.297 | 0.0178 |
| MAIN     | (skipped) | | 0.2192 | 0.104 | 0.0094 |
| PLEDIT   | (skipped) | | 0.2474 | 0.148 | 0.0145 |
| TITLEBAR | (skipped) | | 0.3701 | 0.042 | 0.0054 |

Notes:
- The curriculum-emphasized sliders/strips (VOLUME, BALANCE, PLAYPAUS) are
  genuinely the best — but hit5 0.57–0.76 still fails the 0.90 bar.
- **CBUTTONS** looked near-perfect on old full_mae (0.012) yet hidden hit5 is
  0.15: the model cannot reconstruct an unseen pressed/unpressed state. The
  state-family task is hard and unsolved, not "solved".
- Component-only files (MAIN/TITLEBAR/PLEDIT/POSBAR) are catastrophic under
  random_rect (mae 0.16–0.37) — direct evidence of the missing inpainting
  ability, now correctly *not* mislabelled as state-family tasks (POSBAR
  track/thumb fix).
- A vs B are near-identical on the alternatives files (state_family dominates);
  the aggregate gap (A 0.054 < B 0.065) is the component-only files entering B
  under random_rect.

## Per-skin hidden (Eval B, worst→best)

| skin | hidden_mae | hidden_hit5 | full_mae |
|---|---|---|---|
| the_four_horsemen | 0.1178 | 0.467 | 0.0198 |
| goodgawd | 0.1153 | 0.789 | 0.0272 |
| dragonzv30amp | 0.1136 | 0.147 | 0.0216 |
| tvxq | 0.0906 | 0.205 | 0.0193 |
| blair_razor | 0.0870 | 0.371 | 0.0187 |
| rancid_amp_5 | 0.0733 | 0.649 | 0.0195 |
| aguileramp | 0.0682 | 0.576 | 0.0148 |
| zelda_amp_gold | 0.0540 | 0.655 | 0.0119 |
| minimalistic_black | 0.0529 | 0.585 | 0.0130 |
| a_halo | 0.0424 | 0.474 | 0.0103 |
| infected_fx_gray | 0.0406 | 0.532 | 0.0083 |
| ruki2 | 0.0244 | 0.853 | 0.0034 |
| cyborg | 0.0228 | 0.812 | 0.0052 |
| engraved4_platinum | 0.0160 | 0.835 | 0.0034 |

(Worst-skin ranking differs from the old support-normalized one — goodgawd has
high mae but high hit5, i.e. a few large outliers over many near-perfect pixels.)

## Coverage

- **A:** evaluated 7/11 — BALANCE, CBUTTONS, EQMAIN, MONOSTER, PLAYPAUS,
  SHUFREP, VOLUME. Skipped (no `alternatives`): MAIN, TITLEBAR, PLEDIT, POSBAR.
- **B:** evaluated 11/11 (random_rect makes every file eligible).

## Conclusions

1. The Gate B failure was **not** primarily POSBAR-mask contamination. The
   POSBAR fix is correct and necessary, but the dominant issue is that the
   model never learned general hidden-region reconstruction.
2. Support-normalized metrics were misleading by 3–4×. All future gating must
   use hidden-normalized metrics (now the default loss + eval).
3. The real bottleneck is **random_rect / arbitrary-region inpainting**
   (hidden mae ~0.20), and the component-only files that depend on it. The
   curriculum over-fit the structured state-family frames and learned almost
   no true inpainting.
4. Next training (when approved) should weight random_rect far more heavily
   and be judged on hidden metrics — and the trainer must filter eligible
   files / include a non-state_family mode so component-only files don't crash
   or no-op (the known training caveat).

No training was run for the re-eval above.

## Phase 0 sanity probes (A/B/C) — local only, hidden loss

After fixing the trainer's mask-mix guard (component-only files no longer crash
a state_family-only run), three tiny local probes confirm the hidden-normalized
loss + architecture actually learn inpainting (these are short overfit/smoke
runs, NOT the gated long Kaggle run). All c48, lr 1e-3, sobel 0.25.

**A — can hidden random_rect fall? (one skin, one file, random_rect=1.0)**
- TITLEBAR was a dud probe (near-uniform dark strip → hidden_l1 ≈0.03 floor from
  step 0, nothing to learn).
- MAIN (textured 275×116), 8k steps: hidden_l1 **0.0996 → 0.036**, monotonic, no
  plateau; easy masks reach 0.007. → inpainting is learnable.

**B — all files, one skin, random_rect-heavy (sf0.3/rr0.7), 6k steps**
- overall hidden_l1 **0.0997 → 0.0501**; every file dropped:
  EQMAIN 0.289→0.076, VOLUME 0.091→0.040, POSBAR 0.097→0.049,
  BALANCE 0.059→0.020, PLAYPAUS 0.032→0.013, MONOSTER 0.056→0.029,
  MAIN 0.101→0.077, CBUTTONS 0.139→0.081, SHUFREP 0.114→0.087 (slowest).

**C — 14-skin oracle skin_id, sf0.3/rr0.7, 4k steps, batch 8**
- overall hidden_l1 **0.2368 → 0.1465**, monotonic, every file improving
  (MONOSTER 0.231→0.103, PLAYPAUS 0.253→0.130, POSBAR 0.219→0.140, ...).
- Starts far higher than single-skin (the oracle must also disambiguate 14
  skins) and is nowhere near converged at 4k — but the multi-skin path trains
  cleanly under the new loss.

**Read:** the measurement foundation is sound and the model can learn hidden
reconstruction; the drops are real and monotonic but **slow** at batch=1 / lr
1e-3 / few-k steps. A real run needs a larger batch, more steps, an
rr-heavy (or balanced sf/rr) mix, and must be judged on hidden metrics. Local
GPU OOM'd at batch 14; Kaggle T4 handled batch 12 previously.

## MAIN floor test (strict) — single image overfit to convergence

Question: can the corrected hidden-normalized V7Completer drive ONE textured
MAIN.bmp below the hidden gate under random_rect masks? If not even this
cheapest controlled case clears the bar, a 14-skin Gate B run is premature.

Setup: skin `a_halo_so_bright_it_bleeds` (the highest combined texture std
0.367 + edge density grad 0.145 of the 14 — deliberately non-trivial), one
file MAIN.bmp, random_rect=1.0, c48, hidden loss, sobel 0.25, lr 1e-3,
batch 1 (forced: one item), 30k steps, snapshot every 5k. Eval mask_samples=16.

| snapshot | MAIN hidden_mae | hidden_hit5 | obs_passthrough |
|---|---|---|---|
| 5k  | 0.2063 | 0.221 | 0.0 |
| 10k | 0.0982 | 0.282 | 0.0 |
| 15k | 0.0509 | 0.325 | 0.0 |
| 20k | 0.0405 | 0.370 | 0.0 |
| 25k | 0.0298 | 0.409 | 0.0 |
| **30k** | **0.0246** | **0.455** | 0.0 |

Acceptance was hidden_mae < 0.015 AND hidden_hit5 > 0.90 — **neither met**.

- hidden_mae 0.0246 sits in the pre-registered 0.025–0.04 "do not launch" band;
  still creeping down (−0.005 over the last 5k) but decelerating, extrapolating
  to ~0.02, not <0.015.
- hidden_hit5 0.455 is the decisive blocker: after 30k steps overfitting a
  *single image*, less than half the hidden pixels are within 5/255, rising
  only ~0.04 per 5k and decelerating. 0.90 is effectively unreachable here.
- obs_passthrough = 0 throughout (copy intact; math clean).

The error is bimodal — flat regions near-perfect, edge/detail pixels badly off —
so MAE looks tolerable while hit5 collapses. MAE can keep creeping down while
the high-frequency pixels stay wrong.

### Key conclusion

The current V7Completer is a useful diagnostic baseline, but it is **not an
adequate hidden-region generator**. It learns low-frequency structure but not
pixel-crisp high-frequency detail, even for a one-image MAIN overfit. This is
an architecture/representation ceiling, not a data, curriculum, or
training-budget problem: it appears in the cheapest possible controlled case
(one file, one skin, 30k overfit), so scaling the same recipe (c64, more skins,
longer Kaggle runs) will not fix it.

### Next action: pause for an architecture decision

Do not scale the same U-Net. The next design must explicitly target
high-frequency reconstruction — candidates (to be chosen deliberately, not via
another ad hoc training loop):
- stronger per-pixel coordinate representation (more Fourier bands / learned
  positional codes) so sharp features can be addressed;
- a final-resolution refinement stage on the generated branch;
- a copy / patch / retrieval path from observed pixels (V6-style or attention
  over observed evidence) rather than pure feed-forward synthesis.

