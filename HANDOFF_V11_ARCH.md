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

## Status / ledger
- 256 diverse skins extracted (data_v10_skins256); held16 / train16 / train64
  render sets built (smoke views). s16/s64 scaling done (see HANDOFF_V10_SEARCH).
- NEXT: implement richer-K/V variant in models/bmp_expert_net.py behind a knob
  (buffered for ckpt rebuild), A/B vs avg-pool on held-out. THINK before each
  step; write results here.
