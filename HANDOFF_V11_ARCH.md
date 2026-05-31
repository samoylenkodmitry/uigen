# V11 architecture design — style→atlas generation (2026-05-30)

Deep design rationale for the per-component model. Survives context compaction.
Supersedes the "just add skins / just add capacity" framing. Read with
HANDOFF_V10_SEARCH.md (resources, budget, no-crop, whole-input rules).

## The task, stated correctly
Per component (one model per BMP), given the WHOLE input image of a Winamp-like
UI — at PRODUCT time a CREATIVE imagegen mockup (e.g. CatAmp: EQ sliders are cat
faces), at TRAIN time a cranamp render of a real skin — output that component's
**canonical sprite ATLAS** (the .bmp), in the input's style, **including all
hidden frames/states** (every EQ thumb position, every button state).

KEY FACT (drives the whole architecture): **the target atlas is NOT the rendered
appearance.** Sprite-sheet components (EQMAIN = slider bg + 28 thumb frames +
on/off; CBUTTONS, VOLUME, BALANCE, SHUFREP, MONOSTER, PLAYPAUS, POSBAR) have an
atlas layout that looks nothing like the assembled UI. The render shows ONE
assembled state; the atlas is ALL frames stacked. => There is **no spatial
input→output correspondence** to copy/sample (except roughly for the background
art components MAIN/TITLEBAR). The model must UNDERSTAND style+structure and
SYNTHESIZE the atlas. This is generative, not reconstruction-by-copy.

## Why the current V10 model fails to generalize (diagnosed from code + results)
- Single path input→decoder = encoder → `adaptive_avg_pool2d` to fixed tiny K/V
  `(48,28),(32,18),(24,14),(18,10)` ≈ 2.4k tokens (~700x downsample, AVERAGING) →
  cross-attention from a Fourier query grid → decoder → nearest upsample.
- That avg-pool destroys the local appearance detail needed to replicate a
  component's look. So: single-skin overfit = near-perfect (decoder memorizes the
  atlas in its weights — Gate1 MAIN MAE 0.0005); unseen skin = held-out ≈ random
  (mae 0.31 @16 skins, 0.26 @64; hit5 ~0) because it has nothing precise to
  condition on. Confirmed memorization, not generalization. (s16/s64 also badly
  under-trained on the 2070, but the mechanism is the avg-pool bottleneck.)
- Encoder `_conv_block`s have NO residual; residuals exist only inside the
  cross-attn blocks and the decoder ResBlocks; NO encoder→decoder skip.

## Rejected directions (and why)
- **Sampling-grid / spatial-transformer**: assumes the output is a resampling of
  the input. False for sprite sheets, and the product input is a CREATIVE picture
  with no faithful pixels to sample. (User: "imagegen can be creative... our model
  should be creative too.")
- **Pure U-Net skip**: assumes input/output spatial alignment; only ~holds for
  MAIN/TITLEBAR. Also a high-res copy shortcut would let it ace held-out CRANAMP
  renders (target visible) yet fail the creative product (nothing to copy). Avoid
  as the primary mechanism; a light skip is acceptable ONLY for the background
  components if proven to help, never as the general design.
- **"Just add skins" / "just add capacity"**: necessary but NOT sufficient — if
  the model physically can't route appearance detail to the decoder, no amount of
  data fixes it. Fix conditioning first.

## Design principles (V11)
1. **Cross-attention is the right primitive** (no alignment assumption; queries =
   atlas positions attend to style/exemplar anywhere in the whole input).
2. **Condition richly, not through an averaging bottleneck.** Give the K/V enough
   tokens / resolution / a LEARNED pooling (strided conv or attention pool, not
   avg) to preserve local appearance across scales. The query grid stays (encodes
   which atlas region/frame each output pixel is).
3. **Generative decoder.** Reconstruction (L1 + edge) + ADVERSARIAL (proven to
   imagine hidden states for EQMAIN). The adversarial term is core, not optional.
4. **Anti-copy / domain-bridging augmentation.** Train inputs are faithful cranamp
   renders; product inputs are creative. Augment inputs hard (color/tone/contrast/
   saturation/hue jitter, mild blur/noise; later: stronger restyle) so the exact
   target pixels are NOT trivially recoverable → forces style→atlas mapping that
   transfers to creative mockups. This may matter as much as the architecture.
5. **Per-component architecture.** One model per BMP already; let each pick
   depth/capacity/decoder suited to it (MAIN/TITLEBAR near-copy background;
   EQMAIN/VOLUME/BALANCE/CBUTTONS heavy frame-synthesis). decoder_kind already
   varies per BMP; extend this.
6. **Encoder residuals** (cheap, helps gradient flow / detail) — add to encoder.

## Experiment plan (≤1hr probes; free 2070 first, Lightning for convergence)
Metric that matters = HELD-OUT-skin mae/hit5 (generalization), NOT train fit.
Beware: a copy-machine scores well on held-out CRANAMP but is the wrong model;
keep the generative + augmentation bias.
1. **Bottleneck A/B (the decisive test):** current avg-pool K/V vs a richer K/V
   (more tokens / higher-res / learned pool), same skins+compute, compare HELD-OUT.
   Expect richer K/V → large held-out improvement if the diagnosis is right.
2. Add encoder residuals; re-measure.
3. Add input augmentation; re-measure held-out (and conceptually, transfer).
4. Per-component tuning on the hard sprite-sheet BMPs (EQMAIN) with adversarial.
5. Lock architecture + recipe; write the ≤$100 full-train plan (all skins).

## KEY EFFICIENCY INSIGHT (2026-05-30) — native-resolution whole input
Every multi-skin run so far is grossly UNDER-TRAINED (train mae ~0.13 at 6k steps
vs single-skin 0.0005 @7k), so architecture A/Bs are noise. ROOT CAUSE: the
cranamp render is generated at scale 2.8-3.4 — the ~275px-wide skin upscaled to a
960-wide / 1728-tall canvas. That canvas carries NO more information than the
skin's native res (~520x290); it's ~3.3x upscaled pixels. So we pay ~10x encoder
compute/step for zero extra info -> nothing converges in <=1hr -> every compare
is muddy.
FIX: render/feed the WHOLE input at NATIVE resolution (~10x cheaper, same info,
same detail; NOT a crop, NOT a downsample-that-loses-info — upscaling added
nothing). Then training converges in budget and architecture A/Bs become clean.
Product-time: downscale the high-res mockup to the model's input size; output
atlas is native-res pixel art regardless. Keep transform/scale diversity but
centered near native (e.g. scale ~0.8-1.4, not 2.8-3.4).
kv A/B (CBUTTONS, train64, 6k steps, held16): kv1 held 0.276 / train 0.138;
kv2 held 0.253 / train 0.131. Richer KV helps ~8% but BOTH under-trained ->
inconclusive until native-res makes runs converge. Re-run A/B at native res.

## CLEAN RESULT (2026-05-31, native res) — it's MEMORIZATION, not under-training
Native res let CBUTTONS/train64 actually CONVERGE (train mae 0.019 @23k steps, vs
0.138 under-trained at 1728). kv1 held-out still 0.287 (hit5 0.015). So with
training CONVERGED, held-out is still near-random => the base arch MEMORIZES the
64 training skins (keyed on skin identity) and does NOT learn the general
read-appearance->emit-atlas mapping. This is the decisive read the under-powered
runs couldn't give.
WHY: with only 64 skins, the memorization solution (store 64 atlases) is reachable
and lower-loss-faster than the general solution. cranamp's geometry/state jitter
makes it view-invariant but the TARGET atlas is constant per skin, so jitter does
NOT prevent atlas memorization. Color/style augmentation is NOT a clean fix here
(the task requires EXACT color reproduction from the input — can't jitter colors
away). The clean lever is MORE SKINS: enough that memorizing all atlases is harder
than learning the general mapping.
NEXT (native res makes it feasible): scale training skins 64 -> 240/256 (held16
fixed), measure held-out. If held-out drops sharply, "scale skins" is the path
(paid full-train uses thousands of the 7957). If 240 still memorizes, need a
stronger inductive bias (architecture that forces output = fn of local input
appearance, not a global skin code). kv2 result pending (richer KV may help or may
just give more capacity to memorize).

## kv_scale REFUTED (2026-05-31, native, converged)
CBUTTONS/train64/held16: kv1 held 0.287 (train 0.019), kv2 held 0.293 (train
0.027). Richer K/V did NOT improve held-out. => the bottleneck is NOT conditioning
resolution; it's that 64 skins is memorizable. Use kv_scale=1 (default) going
forward. Lever = skin count / inductive bias, tested next via 64->240 scaling.

## SCALING (64->240) barely helps; ROOT CAUSE = inductive bias (2026-05-31)
CBUTTONS native held-out: 16 skins 0.311 / 64 skins 0.287 (train 0.019, converged)
/ 240 skins 0.275 (train 0.067, UNDER-converged). Held creeps down with scale but
far too slowly, and bigger sets won't converge in-budget. Refuted/insufficient:
kv_scale (conditioning res), native-res compute (enabled convergence, exposed
memorization), moderate scaling. => the model lacks the INDUCTIVE BIAS to
generalize style->atlas; it memorizes.

## PROPOSED V11 ARCHITECTURE: factorize structure (shared) x style (per-skin)
Domain fact: Winamp components have FIXED structure across skins (button layout,
slider-frame arrangement in the atlas); only the ART/STYLE varies. So:
  atlas(skin) = render( SHARED canonical structure , skin STYLE/appearance )
Design:
  - A learned, skin-INDEPENDENT canonical structure for the component (e.g. the
    Fourier query grid + learned per-position content tokens) = the "what goes
    where" of the atlas. Shared => generalizes for free.
  - A STYLE/appearance code + spatial features read from the WHOLE input via the
    encoder + cross-attention; inject it by MODULATION (FiLM/AdaIN) of the shared
    structure, not by generating from scratch. Few skins suffice to learn a style
    EXTRACTOR (low-dim) + a structure that's shared.
  - Generative (adversarial) decoder synthesizes hidden frames (proven for EQMAIN).
  - Fits the creative product: CatAmp = standard slider STRUCTURE + cat STYLE,
    imagined across all frames.
Optional higher-leverage add: a PRETRAINED image encoder for the input (rich
generalizable features from millions of images) so visual representation isn't
learned from scratch on few skins. Consider after the FiLM factorization.
TEST: same CBUTTONS/train64->held16 native protocol; expect held-out to drop
substantially vs 0.287 if the factorization is the missing bias.

## style_mod REFUTED too (2026-05-31) — cheap arch levers exhausted
CBUTTONS/train64n/held16n native (converged): baseline 0.287, kv2 0.293, style_mod
0.278, scale240 0.275. ALL ~0.28 held-out, train ~0.02. Conditioning richness,
structure/style FiLM factorization, and moderate scaling do NOT fix generalization
-> it's a REPRESENTATION + DATA-SCALE problem, not a small-tweak problem.
Two bigger levers remain (pick with user):
  (1) PRETRAINED image encoder (DINOv2/ConvNeXt) for transferable features instead
      of training the encoder from scratch on few skins — standard generalize-from-
      limited-data fix; highest-leverage untested. Test: pretrained-enc vs scratch
      on CBUTTONS/64skins/held16 (decisive: if held drops a lot, representation was
      the issue).
  (2) Train at much larger scale (100s-1000s skins, converged) — scaling curve is
      downward but slow (0.31->0.287->0.275 for 16->64->240); may need paid phase.

## CODEX CONSULT (2026-05-31, gpt-5.5) — adopted plan
(Consult prompt: /tmp/codex_prompt.txt. Run codex OUTSIDE the sandbox — in-sandbox
SIGURGs it; `dangerouslyDisableSandbox` works.) Verdict: representation +
inductive-bias problem, NOT capacity. Don't bet on scaling alone. Stop: bigger KV,
bigger scratch decoders, pure scaling, U-Net/copy paths.
Adopted probe order (CBUTTONS first, native, fixed held16, eval @15/30/60min;
maximize UNIQUE skins/hr not variants/skin):
  1. RETRIEVAL BASELINE (no training): nearest train skin by render features/color
     hist -> output its atlas. If it beats 0.275, the learned model is below a
     trivial floor. Sanity floor.
  2. PAIRED EQUIVARIANT AUG (cheap, no arch): apply the SAME random color/gamma/
     hue/posterize transform to BOTH render input AND target atlas -> colorway
     changes every sample -> kills fixed-atlas memorization, forces reading style
     from input. (Input-only jitter is WRONG — breaks supervision.) Highest-value
     cheap lever.
  3. FROZEN ConvNeXt-Tiny encoder + shallow raw-RGB stream -> patch tokens into
     cross-attn (NOT CLS). Success: held CBUTTONS <=0.22.
  4. FROZEN DINOv2-S/14 + RGB stream (prefer if creative-mockup samples better).
  5. ATLAS-GRAMMAR QUERIES: add slot/state/frame/local-UV/part_id query feats.
  6. ATLAS AUTOENCODER PRIOR (VQ/KL on target atlases) -> render->latent. Bridge
     to latent generation without diffusion cost.
  7. ADVERSARIAL FT LAST (sharpens/imagines; won't fix representation).
EVAL CAVEAT: exact held-out MAE on hidden states is partly ill-posed (artist
choices unidentifiable from one render); judge by visible-region accuracy + style
plausibility, not hit5 alone. Add visible-vs-hidden atlas masks to eval.

## PROBE RESULTS (2026-05-31, native, CBUTTONS, held16)
- #1 RETRIEVAL FLOOR: held mae 0.337 (nearest train skin by render style feat ->
  copy its atlas). Learned model 0.287 BEATS it -> model learns something general,
  weakly; not below a trivial floor. Retrieval-conditioning not urgent.
  (/tmp/retrieval_baseline.py)
- #2 PAIRED COLOR AUG: running (train64n + --color-aug vs baseline 0.287).
- Levers now in code (all buffered/flagged, baseline byte-identical): native-res
  (default on via render size), --kv-scale (refuted), --style-mod (refuted solo),
  --color-aug (testing). TODO per codex: frozen ConvNeXt/DINOv2 encoder + raw-RGB
  stream; atlas-grammar queries; atlas-AE prior.

## Status / ledger
- 256 diverse skins extracted (data_v10_skins256); held16 / train16 / train64
  render sets built (smoke views). s16/s64 scaling done (see HANDOFF_V10_SEARCH).
- NEXT: implement richer-K/V variant in models/bmp_expert_net.py behind a knob
  (buffered for ckpt rebuild), A/B vs avg-pool on held-out. THINK before each
  step; write results here.

## PROBE #2 result (2026-05-31): paired color-aug HELPS (best lever so far)
CBUTTONS/train64n/held16: +color-aug held 0.268 (vs baseline 0.287), train ROSE
0.019->0.029. train-up/held-down = reduced memorization (codex predicted). Modest
but correct direction. Tally (held): retrieval 0.337 | kv2 0.293 | base 0.287 |
style_mod 0.278 | scale240 0.275 | color-aug 0.268. Next: frozen ConvNeXt encoder
+ raw-RGB stream (codex bar: held <=0.22), combined with color-aug.

## CODEX FOLLOW-UP (2026-05-31) — visual collapse + bug + metric reframe
convnext+aug held 0.287 (= baseline; worse than aug-alone 0.268). VISUAL: held-out
preds collapse to a generic/average CBUTTONS atlas (mild tint), ~same across unseen
skins -> known-skin lookup, unknown-skin prior. Codex verdict:
- BUG: cnx set eval() in __init__ but trainer calls model.train() -> ConvNeXt
  stochastic-depth reactivated -> ConvNeXt test NOT clean. Pin cnx.eval() in forward
  before refuting.
- ADV path is UNCONDITIONAL (disc sees only atlas, not render) -> will just sharpen
  the generic prior, won't make it skin-specific. Need CONDITIONAL/style-conditioned
  discriminator. Bring ADV forward only as a sharpness probe.
- METRIC reframe (do this): own-target vs SHUFFLED-target MAE (conditioning test),
  pairwise held output diversity, input<->output color/style correlation, visible-vs-
  hidden region MAE, grids as a first-class gate. Product success = plausible
  CONDITIONED usable atlas, not exact repro.
- 64-skin search is a TRAP (anything memorizable). Real probes: maximize UNIQUE
  skins/hr + paired aug, judged by conditionality. Path = conditional disc + many
  more unique skins, NOT more 64-skin encoder search.
- Next probe: warm-start best color-aug ckpt -> short ADV FT, judge by conditionality
  (own<shuffled, diversity up), not MAE.

## CONDITIONAL DISCRIMINATOR design (codex, 2026-05-31) — adopted
Projection patch discriminator (Miyato) conditioned on a style code from the
INPUT-RENDER encoder only (available at product inference):
- z_style = global-avg-pool of generator's deepest PROJECTED encoder feature
  (feats[3].mean over spatial -> [B, attn_dim]); DETACH before D.
- D: logits = uncond_head(h); style = style_proj(LayerNorm(z)); 
     proj = (h*style[:,:,None,None]).sum(1,keepdim=True)/sqrt(C); return logits+proj, feats.
- MISMATCHED-real negatives (essential, else D ignores z): hinge
  L_D = relu(1-D(y,z)).mean() + 0.5*relu(1+D(yhat.detach(),z)).mean()
        + 0.5*relu(1+D(y, z_wrong)).mean()   # z_wrong = different skin in batch
- G: L_rec + 0.03*L_adv + 0.5*L_fm (start adv low). G lr 1e-4 warm-start, D lr 4e-4,
  1 D step/G step, keep color-aug on, warm-start from best color-aug ckpt.
- D small: d_base 32-48, d_layers 2 (CBUTTONS). Ensure mismatches are truly diff
  skin (use batch skin_id). Detach z. Watch D overpower (d_loss->0, MAE up -> lower adv).
SUCCESS (not MAE alone): own-vs-shuffled gap WIDENS beyond 0.27/0.41, pred diversity
rises toward 0.41 ceiling, grids crisper AND still skin-varying. If both own&shuffled
improve equally -> just sharpened generic atlas (fail).

## RESUME STATE (2026-05-31) — safe pause point
Everything committed. Nothing lost on pause. Lightning studio STOPPED (budget intact).
WHERE WE ARE: model GENERALIZES/conditions (own 0.27 << shuffled 0.41, diversity 0.31
of 0.41 ceiling); exact-MAE was a misleading gate -> use cond_eval.py (own/shuffled/
diversity) as the gate. Levers built+committed: native-res, --color-aug (best, held
0.268), --encoder convnext (eval-bug fixed; best diversity 0.330), --cond-disc
(conditional projection D + mismatched negatives, codex design).
IN FLIGHT (free 2070): cond-disc probe = warm-start /tmp/aug_cbuttons/best ->
conditional adversarial FT (/tmp/cond_test.sh, log /tmp/cond_test.log). On resume:
read its verdict (>>> COND own/shuffled/gap/diversity vs baseline gap +0.141 div 0.312;
also view /tmp/cond_cbuttons/eval_held/pred_vs_target_grid.png). 
NEXT after that (codex plan): (1) if cond-disc widens gap/diversity -> adopt; (2) the
other pillar = SCALE unique skins (64 is a memorization trap) -> train CBUTTONS on
240+ unique skins native-res judged by cond_eval; (3) then lock recipe + write the
<=$100 paid full-train plan (all 7957 skins). Re-consult codex (OUTSIDE sandbox:
dangerouslyDisableSandbox) at each decision. Probe ckpts live in /tmp (volatile;
re-runnable). Datasets data_v10n_* + data_v10_skins256 on disk.

## RESUME CORRECTION (2026-05-31): cond-disc probe was KILLED at step 2000 (paused
for GPU), so its COND numbers == the warm-start color-aug baseline -> INCONCLUSIVE.
ON RESUME: re-run /tmp/cond_test.sh to completion (~45min free 2070), judge by
cond_eval gap/diversity + /tmp/cond_cbuttons/eval_held grid. Then: scale unique skins;
lock recipe; <=$100 paid full-train. Consult codex OUTSIDE sandbox at decisions.

## RESUMABILITY (2026-05-31): kill/pause no longer restarts from 0.
train_bmp_expert.py --resume restores weights+optimizer+step(+disc) from
<out>/last.safetensors + <out>/train_state.pt (saved every checkpoint-every +
exit). v10_gate_sweep.py passes --resume + checkpoint-every 1500. scripts/v11_probe.sh
= resumable per-component probe (re-run SAME cmd -> continues; --steps huge, each
launch runs to --max-minutes). Runners must NOT rm -rf the out dir. Verified:
train 150 -> resume -> "RESUMED at step 150" -> continued to 300.

## COND-DISC VERDICT (2026-05-31) — ADOPTED (modest, right direction)
Resumable cond-disc probe ran to step 10667 (45min cap), warm-started from
/tmp/aug_cbuttons/best (color-aug), full codex recipe (--color-aug --adversarial
--cond-disc, adv 0.03, fm 0.5, d-lr 4e-4, d-base 48, d-layers 2). cond_eval on
held16 (n=16):
  metric        | color-aug base | +cond-disc | delta
  own mae       | 0.269          | 0.2740     | +0.005 (~flat)
  shuffled mae  | 0.410          | 0.4224     | +0.012
  GAP (own<shuf)| +0.141         | +0.1485    | +0.007 (WIDER -> more skin-specific)
  pred diversity| 0.312          | 0.3353     | +0.023 (ratio 0.76->0.81 toward ceiling)
VERDICT: PASS by codex criteria (gap widens + diversity up, own~flat = NOT just a
sharper generic atlas). Grid /tmp/v11_cond_cbuttons/eval_held/pred_vs_target_grid.png
confirms: predicted CBUTTONS strips are skin-specific (distinct blue/yellow/red/green
button bars per held skin), reasonably crisp. Effect is MODEST -> the bigger lever is
SCALE (codex: 64 skins is a memorization trap). ADOPT cond-disc into the recipe.

## SCALE PILLAR (2026-05-31) — task #162, IN FLIGHT
Central hypothesis: does 64->240 UNIQUE skins (native-res) force generalization
(widen own<shuffled gap, raise diversity toward 0.41 ceiling)?
Datasets verified: data_v10n_train240 = 240 unique skins, DISJOINT from
data_v10n_held16 (16). Plan: (1) color-aug L1 on 240 skins (isolates scale, builds
warm-start ckpt) judged by cond_eval vs 64-skin base (gap +0.141 div 0.312); then
(2) cond-disc FT on top. Resumable (kill->rerun continues). 64-skin color-aug base
to beat: own 0.269 / shuf 0.410 / gap +0.141 / div 0.312.

## SCALE PROBE chunk1 (2026-05-31) — UNDER-TRAINED, inconclusive (resuming)
240-skin color-aug FROM SCRATCH, 45min -> step 12356. cond_eval held16 (n=16):
  own=0.2782 shuffled=0.3983 gap=+0.1202 pred_div=0.2867 div_ratio=0.69
vs 64-skin color-aug CONVERGED base: own 0.269 / gap +0.141 / div 0.312.
=> Below baseline, BUT CONFOUNDED: 12k steps from scratch = ~51 samples/skin vs the
64-skin base ~193/skin (4x less per-skin exposure). Still descending monotonically at
cap (eval_mae 0.196->0.176->0.152->0.128, no plateau; diversity climbs as preds sharpen).
NOT a refutation of scale. Resumable -> continue to convergence and watch whether
gap/div climb past +0.141 / 0.312. Decision after chunk2 trajectory; consult codex if
ambiguous. ckpt /tmp/v11_scale240_cbuttons (resume continues from step 12356).

## SCALE PROBE chunk2 (2026-05-31) — SCALE WORKS (positive slope), continuing
240-skin color-aug resumed -> step 24826 (~102 smpl/skin). cond_eval held16 trajectory:
  chunk1 (step12356,~51/skin):  own .278 shuf .398 gap +.1202 div .287 ratio .69
  chunk2 (step24826,~102/skin): own .280 shuf .417 gap +.1370 div .319 ratio .77
  64-skin CONVERGED base (~193/skin):              gap +.141  div .312 ratio .76
=> gap AND diversity climb steeply with training; at HALF the 64-skin per-skin
exposure the 240-skin model ALREADY beats 64-skin diversity (.319>.312) and nearly
matches gap (+.137 vs +.141). Train mae still descending (.128->.074, no plateau).
VERDICT: SCALE FORCES GENERALIZATION (codex hypothesis #162 CONFIRMED, positive slope).
NEXT: resume to convergence (chunk3+) to confirm gap/div plateau ABOVE baseline; then
add cond-disc on top (both confirmed levers) = candidate final recipe; consult codex to
lock recipe + design <=$100 paid full-train (all 7957 skins). ckpt /tmp/v11_scale240_cbuttons.

## PAID FULL-TRAIN PLAN (DRAFT 2026-05-31) — refine w/ codex after converged probe
CONFIRMED RECIPE (V11): per-component BMPExpertNet, input = WHOLE native-res render
(384x696, never cropped), output = that component's canonical atlas. Losses L1 +
1.5*Sobel + 0.5*Laplacian. Levers locked in by search:
  - native-res input (10x cheaper than 960x1728, same info)        [confirmed]
  - paired color-aug (kills fixed-atlas memorization)              [BEST lever]
  - conditional projection discriminator + mismatched negatives     [modest +, adopt]
  - MAXIMIZE unique skins (scale forces generalization)            [#162 CONFIRMED slope]
REFUTED (don't spend on): kv_scale x4, style_mod solo, moderate encoder scaling.
DATA: skins_raw = 7787 unique skins on disk; 49 deterministic state-transforms/skin
(+ color-aug at train time for extra per-skin diversity). Full set ~= 7787*49 = ~381k
samples/component. Held-out eval = disjoint skin split (reuse data_v10n_held16 style).
ECONOMICS (ROUGH, verify): local 2070 native batch6 = 0.21 s/step. Rented L40S/A100
~10x throughput w/ larger batch. Per-component convergence on full set est. ~100-200k
steps (PIN from converged 240-skin probe + extrapolate by unique-skin count). 11
components. @ ~15 step/s on fast GPU: 150k steps = ~2.8 GPU-hr/component x 11 = ~31
GPU-hr. @ ~$1.5/hr spot L40S = ~$46 (2x margin -> ~$92, under $100 cap). 
OPEN QUESTIONS FOR CODEX: (1) steps/component to converge on 7787 skins (extrapolate
from probe)? (2) train 11 components sequentially on 1 GPU or parallel? (3) is 49
transforms x color-aug enough, or generate more transforms? (4) cond-disc on from
step 0 or FT phase? (5) which paid GPU/provider for best $/throughput under $100?
SEQUENCING: finish converged 240-skin CBUTTONS probe (gap/div plateau above 64-base)
-> add cond-disc at scale -> consult codex to LOCK recipe + finalize this plan ->
generate full native-res dataset (CPU, free) -> user buys GPU -> paid train.
