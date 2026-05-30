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

## Status / ledger
- 256 diverse skins extracted (data_v10_skins256); held16 / train16 / train64
  render sets built (smoke views). s16/s64 scaling done (see HANDOFF_V10_SEARCH).
- NEXT: implement richer-K/V variant in models/bmp_expert_net.py behind a knob
  (buffered for ckpt rebuild), A/B vs avg-pool on held-out. THINK before each
  step; write results here.
