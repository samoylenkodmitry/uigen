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

## SCALE PROBE chunk3 (2026-05-31) — SCALE CONFIRMED + ceiling identified
240-skin color-aug -> step 37372 (~153 smpl/skin). cond_eval held16 full trajectory:
  c1 step12356 ~51/skin:  own .278 shuf .398 gap +.120 div .287 ratio .69
  c2 step24826 ~102/skin: own .280 shuf .417 gap +.137 div .319 ratio .77
  c3 step37372 ~153/skin: own .271 shuf .414 gap +.144 div .313 ratio .76
  64-skin CONVERGED ~193/skin: own .269 shuf .410 gap +.141 div .312 ratio .76
FINDINGS: (1) SCALE GENERALIZES w/ NO memorization penalty — 240 skins MATCHES/beats
the 64-skin converged baseline on ALL conditioning metrics at LESS per-skin exposure
(=> safe to use all 7787 skins). (2) But conditioning PLATEAUS at an architecture-set
ceiling (gap ~.14, div ~.31, ratio stuck ~.76); scale alone didn't push div past it
(c2->c3 div flattened .319->.313). => cond-disc is the lever for diversity/crispness
(it took 64-skin div .312->.335). CANDIDATE FINAL RECIPE = SYNTHESIS scale + cond-disc.
NEXT: cond-disc FT warm-started from this 240-skin ckpt (does div push past .335 toward
.41 ceiling at scale?) + consult codex (full trajectory, diversity-ceiling question,
paid-train plan review). ckpt /tmp/v11_scale240_cbuttons.

## CODEX CONSULT #4 (2026-05-31) — ceiling diagnosis + paid plan + lock criteria
Q1 PLATEAU = architecture/objective ceiling (NOT under-train; train mae drops while div
flattens). TOP 2 LEVERS: (1) SPATIAL conditional D — cond-D today uses only GLOBAL pooled
z_style (bmp_expert_net.py ~254); make D patch features cross-attend to encoder style
tokens OR dot against a style map at D-logit resolution; ALSO make z_wrong skin-id-safe
(not plain roll, train_bmp_expert.py ~311). (2) ATLAS-GRAMMAR query feats (slot/state/
local-UV labels; cheap for CBUTTONS). NOT: diffusion/VAE head (inflates div w/o
faithfulness), VGG perceptual (eval-only at best).
Q2 METRIC: cond_eval = right SEARCH gate, not product north-star. Cheap upgrades:
all-pairs diversity + bootstrap CI; **visible-render ROUND-TRIP** (insert predicted
component into real held skin -> render_visible (atlas_ai/torch_cranamp_renderer.py:179)
-> compare masked visible px to held render); for mockups render(pred skin) vs mockup
RGB+Sobel+DINO/LPIPS over chrome masks.
Q3 PAID PLAN: don't extrapolate steps/skin literally; 240-run ~= 19 epochs. Full =
7787*49 ~= 381k rows; batch48 ~= 8k steps/epoch. Per component: L1+coloraug 8-12 epochs
(64-95k steps) + cond-disc FT 1-3 epochs (8-24k); hard (EQMAIN/VOLUME/BALANCE/PLEDIT)
1.5x -> 120-180k. SEQUENTIAL on 1 GPU (parallel fights mem/IO); 1 component/GPU if many.
49 transforms ENOUGH (more unique skins > more variants). Cond-disc = FT not step0.
GPU: RunPod L40S 48GB $0.86/hr or RTX6000Ada $0.77/hr (top picks); A100 SXM $1.49;
Hyperstack A100 $1.35 / L40 $1.00 / A6000 $0.50; Vast cheaper but babysit. => <<$100.
Q4 LAST PROBES BEFORE LOCK: (1) finish 240 cond-disc FT [GO if div>=.345 gap>=.150
own<=.285; strong if div>=.36]. (2) SPATIAL cond-D probe, warm-start 240 ckpt 45-60min
[GO if beats global cond-D by +0.02 div & +0.01 gap, own not worse >.01]. (3) HARD-
component smoke EQMAIN or VOLUME same recipe 45-60min [GO if no OOM, descends, cond_eval
non-generic]. "Near lock; not fully locked until 240 cond-disc + one hard component pass."

## PROBE (1) GLOBAL cond-disc AT SCALE (2026-05-31) — modest, BELOW bar
240-skin scale ckpt -> global cond-disc FT (adv .03, fm .5, d-lr 4e-4, d-base48 d-layers2)
-> step 12276. cond_eval held16: own .2696 shuf .4145 gap +.1449 div .3206 ratio .78.
vs 240 scale base (pre-cond): gap +.144 div .313. => only +.008 div (global cond-disc
helps LESS at 240 than at 64: +.008 vs +.023). MISSES codex go-bar (div>=.345 gap>=.150).
Confirms GLOBAL conditioning ceiling -> motivates SPATIAL cond-D.
ckpt /tmp/v11_scale240_cond_cbuttons.

## PROBE (2) SPATIAL cond-D AT SCALE (2026-05-31) — IN FLIGHT
Implemented (commit 34bb14f): D patches cross-attend over render style TOKENS + skin-id-
safe mismatched negative. Clean A/B: SAME 240-skin warm-start as probe(1), --spatial-cond-d.
Codex go-bar: beat global by +.02 div & +.01 gap, own not worse >.01 => target div>=.341
gap>=.155 own<=.280. ckpt /tmp/v11_scale240_spatcond_cbuttons.

## PROBE (2) SPATIAL cond-D VERDICT (2026-05-31) — REFUTED (no win)
Spatial cond-D, same 240 warm-start, step 11881: own .2744 shuf .4154 gap +.1410
div .3120 ratio .76. A/B at equal warm-start+budget:
  240 base (no cond):    own .271 gap +.144 div .313 ratio .76
  (1) GLOBAL cond-disc:  own .270 gap +.145 div .321 ratio .78  <- best
  (2) SPATIAL cond-D:    own .274 gap +.141 div .312 ratio .76  <- = base, < global
=> SPATIAL cond-D REFUTED as quick win (below codex +.02 div bar; ~= no-cond base).
Diversity CEILING (~.31-.32, ratio ~.76) sticky: neither global nor spatial conditioning
breaks it much at scale. GLOBAL cond-disc stays the (modest) best conditioning lever.
ckpt /tmp/v11_scale240_spatcond_cbuttons. Code kept (--spatial-cond-d, opt-in, off by default).

## PROBE (3) HARD-COMPONENT SMOKE EQMAIN (2026-05-31) — IN FLIGHT
EQMAIN (275x116, hardest: 1px slider grooves) L1+color-aug on 240 skins native-res, to
de-risk paid budget: throughput / OOM / does it descend / cond_eval non-generic (codex
lock gate 3). ckpt /tmp/v11_eqmain240.

## CODEX CONSULT #5 (2026-05-31) — LOCK THE RECIPE
DECISION: LOCK. Stop chasing div_ratio (it's a PROXY ceiling — small held set, adjacent-
pair diversity, hidden atlas px not fully determined by one visible render; product
success = conditioned VISIBLE plausibility, not exact hidden-atlas diversity). Keep
div_ratio ONLY as a regression alarm (CBUTTONS fail if div<.29 | gap<+.13 | own>.285).

### >>> LOCKED V11 RECIPE <<<
native-res input (384x696, NEVER cropped) + paired COLOR-AUG + GLOBAL cond-disc FINE-TUNE
(warm-start from L1+edge+color-aug; Miyato projection + mismatched-real negatives; NOT
from step 0) + MAX UNIQUE SKINS. Model = BMPExpertNet (conv enc -> pooled KV -> Fourier-
query cross-attn -> progressive conv dec -> sigmoid). Loss L1 + 1.5*Sobel + 0.5*Laplacian.
DROP spatial cond-D (refuted for this budget; code kept opt-in/off via --spatial-cond-d).

### LAST go/no-go before paid train = CBUTTONS ROUND-TRIP sanity (NOT a search loop)
Minimal (codex Q2): insert predicted CBUTTONS into a held skin's REAL target BMPs ->
render_visible (atlas_ai/torch_cranamp_renderer.py) -> compare to an ORACLE proxy render
built from the SAME held target BMPs (NOT the randomized held render — that's confounded).
Mask = the 6 visible transport-button rects. LOCK if: rt_gap>=+0.08 OR own<=0.75*shuffled;
global cond-disc no worse than base by >.01 visible MAE; grid has no systematic wrong-
color/style buttons. Build for CBUTTONS now; EQMAIN next IF its smoke ckpt exists.
Q4: spatial cond-D = refuted operationally; no more search shots on it.

## ROUND-TRIP SANITY = PASS + EQMAIN SMOKE = PASS (2026-05-31) — SEARCH COMPLETE
scripts/roundtrip_eval.py (CBUTTONS, visible 6-button rects, oracle=render of real held
targets, own vs shuffled): own_vis .2677 shuf_vis .3485 rt_gap +.0808 own/shuf .768
LOCK_PASS=True; cond-disc vs base delta -.0011 (cond-disc >= base). Grid: predicted
buttons match each skin's color/material/palette (no systematic wrong-color); glyphs soft
(L1 sharpness limit codex said NOT to chase; paid train's 64-95k steps + cond-disc sharpen).
EQMAIN smoke (hardest, 275x315): OOMs at batch6 FP32 on 8GB 2070 -> batch2 trains fine
(no OOM, mae descends .233->...), de-risking the PAID budget (L40S 48GB handles batch6+).
=> ALL 5 LOCK GATES MET. V11 ARCHITECTURE SEARCH COMPLETE. Recipe LOCKED (see CONSULT #5).
Free-GPU search hours spent on V11: well under the 70-hr cap. Next = PAID full-train (user
buys GPU). See "PAID FULL-TRAIN PLAN" + finalized RUNBOOK below.

## >>> PAID FULL-TRAIN RUNBOOK (finalized 2026-05-31) <<<
Recipe LOCKED (consult #5). User buys GPU; these are the exact steps.
STEP 0 (free, optional pre-gen): generate full native-res dataset on the rented box (or
local CPU) — scripts/make_v10_bmp_expert_dataset.py over ALL 7787 skins in skins_raw/,
--canvas-w 384 (native; height auto), 49 deterministic transforms/skin (+ train-time
color-aug for extra per-skin diversity). ~7787*49 ~= 381k renders/component-CSV. Reserve
a DISJOINT held-skin split (~64 skins) for eval. (~3-4 CPU-hr, parallelizable.)
STEP 1 (paid GPU, sequential per component, 11 total): for each BMP, on RunPod L40S 48GB
($0.86/hr) or RTX6000Ada ($0.77/hr), batch 48:
  (a) L1+edge+COLOR-AUG: ~8-12 epochs (~64-95k steps; hard EQMAIN/VOLUME/BALANCE/PLEDIT
      1.5x -> 120-180k). cmd = train_bmp_expert.py --color-aug --amp --batch 48 --resume
      --checkpoint-every 2000 --progress-every 200 (NEVER crop input; native-res data).
  (b) GLOBAL cond-disc FT from (a) best: ~1-3 epochs (~8-24k steps). add --adversarial
      --cond-disc --adv-weight .03 --fm-weight .5 --d-lr 4e-4 --d-base 48 --d-layers 2
      --init-from <a/best> --lr 1e-4. (NO --spatial-cond-d: refuted.)
  Judge each on the DISJOINT held split: cond_eval (regression alarm: div>=.29 gap>=+.13
  own<=.285) + roundtrip_eval.py (rt_gap>=+.08) where wired. Resumable: forced stop loses
  <=2000 steps.
BUDGET: ~31-45 GPU-hr total @ ~$0.86/hr = ~$27-39 (<< $100 cap; 2x margin still under).
STEP 2: infer_v10.py packages the 11 experts' outputs -> loadable skin.wsz.
STEP 3 (Gate 4 product): feed a CREATIVE imagegen mockup (KITTENAMP-style PNG) through
all 11 experts -> render_visible / pack -> eyeball product quality. (User drops PNG.)
NON-NEGOTIABLES (unchanged): never crop input; native-res; max unique skins; cond-disc
as FT not step0; spatial cond-D OFF.

## EQMAIN SMOKE COND (2026-05-31) — strong bonus de-risk
EQMAIN L1+color-aug, batch2, only step 6258, NO cond-disc yet: own .2421 shuf .4139
gap +.1718 pred_div .3458 tgt_div .3891 div_ratio .89. => conditions BETTER than CBUTTONS
(gap +.144 ratio .76) even under-trained — the EQ window bg+sliders carry strong skin-
specific style. Recipe transfers to the HARDEST component; trains w/o OOM at batch2 on
8GB (batch48 fine on rented 24GB+). Reinforces the lock. ckpt /tmp/v11_eqmain240.

## LIGHTNING SMOKE — REAL THROUGHPUT NUMBERS (2026-05-31)
Studio scratch-studio-devbox (teamspace gpu-model-development-project, AWS cluster).
Env: system python, torch 2.8.0+cu128, torchvision, PIL, numpy, safetensors ALL present
(no venv build). Repo pulled to current. skins_raw was EMPTY on studio -> uploaded an
80-skin tar slice (SDK per-file upload 501s on filenames with parens -> use TAR + extract).
Gen: extract_wsz_skins (62/80 accepted) -> make_v10_bmp_expert_dataset gate1 canvas-w 384
(24 skins, 6160 CBUTTONS rows). MEASURED (CBUTTONS, AMP, native-res 384x696):
  T4 16GB:   batch 16 (32 OOMs!) -> 0.61 s/step = 26.2 samples/s. $0.19/hr (AWS).
  L40S 48GB: batch 48 -> 0.54 s/step = 88.9 samples/s. $2.89/hr od / $1.91 interruptible.
KEY FINDING: run is DATA-LOADING-BOUND (L40S only 3.4x T4, not ~6-8x) — PIL-decoding
384x696 PNGs in the dataloader is the ceiling. Codex's ~$27-39 assumed ~460 samples/s
(compute-bound); reality is ~89 s/s UNTIL we optimize data loading (pre-pack renders to
.pt/webdataset, more workers, RAM cache) -> ~2.5x throughput, big GPUs then cost-effective.
T4 16GB caps at batch 16 for CBUTTONS (and won't fit EQMAIN well) -> 24GB+ (L4/A100/L40S)
needed for batch 48. Cheaper GPUs ARE reachable via other clusters: cloud_accounts incl
gcp-lightning-public-prod (L4 $0.48, A100-80 $2.71) + lightning-lambda-prod (A100-40 $1.29);
A100-40 NOT on the AWS cluster (switch_machine 400).
COST EST (balanced: all 7787 skins x16 transforms x10 epochs x11 comp +25% hard = ~17M passes):
  current pipeline: T4 ~180h~$34 (7d); L4 ~94h~$45 (4d); A100-40 ~47h~$61 (2d); L40S ~53h~$101 (bad value).
  +data-pipeline fix (~2.5x): A100-40 ~21h~$27 (1d); L4 ~38h~$19 (1.6d); L40S ~21h~$40.
RECOMMEND: optimize data loading FIRST (cheap CPU pre-pack), then run on L4 (GCP, cheapest,
~$19-45) or A100-40 (Lambda, ~1-day turnaround, ~$27-61). All inside $100; free 7.27 credits
alone won't cover a full-quality run (a lean 8-transform/6-epoch run ~$5-15 could ~fit free).
Smoke total spend ~ $0.30. Studio STOPPED.

## LEAN BUILD + REAL BENCHMARKS (2026-05-31) — dataset READY on studio
Lean dataset GENERATED on studio: data_v11_lean = 6191 train skins x17 = 105,247 renders
(+ data_v11_lean_held 64 skins x17 = 1088), ALL prepacked to renders_npy/. Persists on
scratch-studio-devbox filesystem (AWS cluster). All clusters reachable incl L4 (GCP)!
MEASURED throughput (CBUTTONS, native-res, AMP, --fast-renders):
  T4 16GB  batch16: 0.62 s/step = 26 s/s ($0.19/hr)  [compute-bound; fast-renders no help]
  L4 24GB  batch32: 0.97 s/step = 33 s/s ($0.48/hr)  [GPU util 98-100% = COMPUTE-bound]
  L40S 48GB batch48: 0.54 s/step = 89 s/s ($1.91 int/$2.89)  [batch48 OOMs L4 24GB]
=> Model is COMPUTE-bound at native-res (no data-pipeline fix helps further). $/sample:
T4 best value (cheap+efficient) but 16GB caps big atlases (EQMAIN 275x315 etc) to small
batch; L4 fits all at batch 32; L40S fastest.
REAL LEAN COST (6191 skins x17, ~5 epochs incl cond-disc, 11 comp +25% hard = ~7.2M passes):
  L4 @33 s/s: ~61 GPU-hr ~ $29 (~2.5d).  T4 @26 (small comps only): ~$15 (~3d).
  => original ~$5-15 was optimistic (assumed ~125 s/s). HONEST lean = ~$15-29.
NEXT: confirm spend/config with user, then train. Studio STOPPED.

## LEAN TRAIN RUNNING + CATAMP TEST WIRED (2026-05-31)
RUNNING on L4 (studio scratch-studio-devbox): scripts/train_all_lean.py --data data_v11_lean
--held data_v11_lean_held --out runs/v11_lean --budget-min 540 (HARD 9h cap; ~$5-6 inside
the 7.27 free credits; auto_shutdown ON as safety). L1+color-aug+fast-renders, resumable,
per-atlas batch (EQMAIN b8/mid b16/small b32), per-component cond_eval on held. Poller
bnboycrfe notifies at DONE. Log: ~/uigen/train_lean.log on studio.
ON TRAIN DONE: (1) stop studio; (2) ckpts at runs/v11_lean/<STEM>/best.safetensors;
(3) CATAMP product test (user request) = infer_v10.py --image eval_mockups/caat.png
(the KITTENAMP cat mockup) --checkpoints runs/v11_lean --canvas-w 384 --canvas-h 696
--out runs/v11_lean/infer_catamp  (NATIVE-RES flags wired commit 069b8e1; prefers
best.safetensors) -> skin.wsz + real-Cranamp render; (4) round-trip eval + show user.
NOTE: lean = free-tier (~1 epoch over 6191 skins) -> SOFT fidelity expected; this is the
free first product look, not the final quality (paid run = more epochs + cond-disc).
