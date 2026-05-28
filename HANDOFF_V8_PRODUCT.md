# V8 Product Pipeline Handoff

## Superseded By V10

This document is retained as history only.

As of 2026-05-28, V8 crop-based visible extraction is **not** the product
architecture. The screenshot comparison showed that V8 is a blind crop-and-paste
baseline: it can produce loadable skins, but it does not learn to semantically
fit buttons, sliders, playlist chrome, and dynamic content into the correct
Winamp assets.

Treat V8 as:

```text
loadability smoke test
renderer/eval infrastructure
deterministic hidden-state compiler prototype
```

Do not present V8 crop outputs as evidence that the mockup-to-skin product works.
The active direction is now:

```text
HANDOFF_V10_BMP_EXPERTS.md
```

The next primary work is V10 BMP experts:

```text
full distorted/generated render -> one expert per output BMP -> final BMP pixels
```

Do not continue V8 crop-baseline polishing as product work.

Date: 2026-05-27

## Historical V8 Gate

The product gate is the **real Cranamp render of the generated `.wsz`**.

`atlas_ai/torch_cranamp_renderer.py` / `render_visible` remains useful as a
differentiable optimization proxy, but it is not allowed to decide product
progress by itself. It can flatter the result because it re-pastes extracted
mockup crops at the expected positions. Every product eval must save and score
both:

```text
render_lite.png          # Cranamp-lite proxy
render_cranamp.png       # real Cranamp CLI render of the generated .wsz
side_by_side_lite.png
side_by_side_cranamp.png
```

Summary metrics must distinguish:

```text
lite_rgb_mae
lite_sobel_mae
cranamp_rgb_mae
cranamp_sobel_mae
load_success
```

Only the `cranamp_*` metrics plus visual inspection are the product gate.

Local history currently has:

```text
9693be7 V8 eval: score product set with real Cranamp renderer (product gate)
4316b1e V8 EQ: write real Cranamp EQ sprite geometry
```

These are valid to push if honest and green. The old proxy-only EQ commit must
not be pushed as product success:

```text
d7f4e8e V8 product baseline + EQ widget extraction fix
```

## Historical V8 Next Cycle: No Longer Primary

This was the next cycle before the later resets. Do not treat it as the current
primary plan unless V10 product eval later proves this specific sprite-geometry
work is the blocker again.

The comparison image shows the next blocker is **sprite-geometry
reconstruction**, not render-random state confound.

Observed real-Cranamp failures:

```text
1. Playlist title is garbled ("PLAYLAYLAYLIST") and right-edge text repeats.
   That means PLEDIT.bmp is being tiled/sampled by real Cranamp from the wrong
   source pixels. Whole-playlist crop into PLEDIT.bmp cannot work.

2. Main window has sprite alignment/composition defects; this is not merely a
   random-state mismatch.

3. EQ fix is real progress, but scoped: it proves the method of reading real
   Cranamp geometry and writing the actual sampled rects.
```

Next work:

```text
1. Add per-component product metrics from real Cranamp renders:
   main_cranamp_rgb_mae / main_cranamp_sobel_mae
   eq_cranamp_rgb_mae / eq_cranamp_sobel_mae
   playlist_cranamp_rgb_mae / playlist_cranamp_sobel_mae

2. Save cropped side-by-sides per component.

3. Read/pin real Cranamp render_playlist geometry.

4. Reconstruct PLEDIT.bmp sub-sprites:
   titlebar left/middle/repeat/right
   actual playlist title/text area
   side borders
   footer pieces
   scrollbar thumb
   selected-row/list background only if sampled from assets
```

Rule:

```text
Whole-window crop works only for areas Cranamp draws as one large bitmap.
Any tiled/sprite-composited area must be compiled into the exact sampled sprite
rects the real engine reads.
```

## Reset

Stop using hidden historical atlas recovery as the main decision metric.

The real product is:

```text
generated Winamp-like mockup image
-> valid Cranamp/Winamp skin files
-> rendered skin looks as close as possible to the mockup
```

The old V7/V7.1 research remains useful background, but it should not drive the
next runs. We learned that padded-atlas metrics lied, hidden states are
underdetermined from one render, and global generators smear fine texture. The
next work must be product-render driven.

## Stop As Primary Work

Do not spend run quota on these unless a V8 product eval proves they are the
current blocker:

```text
1. More generic StateFamilyExpander S2/S3 variants.
2. More hidden-state historical target matching.
3. More full hidden atlas MAE experiments.
4. Bigger V5/V6/V7 runs without a render-match product gate.
```

## New Architecture

```text
V8 = Visible extractor + deterministic hidden-state compiler + render-match refinement
```

Pipeline:

```text
mockup image
-> normalized 960x1728 view + layout.json
-> visible/default exported BMP tensors
-> deterministic plausible hidden states
-> skin.wsz
-> Cranamp-lite proxy render
-> real Cranamp product render
-> optional render-match refinement
-> product eval side-by-side
```

Primary metric:

```text
Does the generated skin load?
Does the rendered output look like the mockup?
Are visible panels/buttons/sliders sharp and in the right place?
```

Automated metrics:

```text
cranamp_rgb_mae
cranamp_sobel_mae
component cranamp MAE/Sobel
lite_rgb_mae / lite_sobel_mae for proxy diagnostics only
load_success
edge/sharpness diagnostics
```

Human visual inspection is required. There is no true atlas for a creative
mockup.

## Current Codex Implementation

New product-path files added in the current worktree:

```text
atlas_ai/v8_layout.py
atlas_ai/v8_assets.py
atlas_ai/visible_extractor.py
atlas_ai/hidden_state_compiler.py
atlas_ai/torch_cranamp_renderer.py
models/visible_skin_net.py
scripts/normalize_mockup.py
scripts/v8_mockup_to_skin.py
scripts/refine_to_mockup.py
scripts/eval_product_set.py
tests/test_v8_product_pipeline.py
toclaude.md
HANDOFF_V8_PRODUCT.md
```

Known unrelated local changes were already present and should not be treated as
part of V8:

```text
scripts/kaggle_live_log.py
scripts/kaggle_set_token.sh
```

## Baseline Commands

Normalize only:

```bash
python scripts/normalize_mockup.py \
  --input eval_mockups/goose.png \
  --out runs/v8/goose_norm
```

End-to-end deterministic product baseline:

```bash
python scripts/v8_mockup_to_skin.py \
  --input eval_mockups/goose.png \
  --out runs/v8/goose
```

Important outputs:

```text
runs/v8/goose/normalized.png
runs/v8/goose/layout.json
runs/v8/goose/debug_overlay.png
runs/v8/goose/render_preview.png
runs/v8/goose/side_by_side.png
runs/v8/goose/skin/skin.wsz
```

Render with existing Cranamp CLI as the product check:

```bash
cranamp_cli/cranamp-cli render-random \
  --skin-dir runs/v8/goose/skin/skin.wsz \
  --seed 7 \
  --canvas-w 960 \
  --canvas-h 1728 \
  --out-view runs/v8/goose/cranamp_render.png
```

Optional render-match refinement:

```bash
python scripts/refine_to_mockup.py \
  --target runs/v8/goose/normalized.png \
  --layout-json runs/v8/goose/layout.json \
  --skin-dir runs/v8/goose/skin \
  --out runs/v8/goose_refined \
  --steps 100
```

Run product set:

```bash
python scripts/eval_product_set.py \
  --mockups eval_mockups \
  --out runs/v8/product_eval_001
```

The product eval writes:

```text
runs/v8/product_eval_001/summary.csv
runs/v8/product_eval_001/summary.json
runs/v8/product_eval_001/<case>/side_by_side.png
runs/v8/product_eval_001/<case>/skin/skin.wsz
```

`summary.csv` includes blank fields for:

```text
human_similarity_1_5
human_sharpness_1_5
notes
```

Fill those after visual inspection.

## Manual Layout Overrides

The current default layout assumes classic windows stacked full width. For real
mockups, use manual rect overrides first. Do not block on auto-detection.

Override JSON:

```json
{
  "rects": {
    "main": [x, y, w, h],
    "eq": [x, y, w, h],
    "playlist": [x, y, w, h]
  }
}
```

Use it:

```bash
python scripts/v8_mockup_to_skin.py \
  --input eval_mockups/goose.png \
  --rects-json eval_mockups/goose_rects.json \
  --out runs/v8/goose_manual
```

Coordinates default to the original input image space. Use
`--rects-space normalized` if the JSON already references the normalized canvas.

## What Each Module Does

`atlas_ai/v8_layout.py`

```text
Letterboxes input images, builds layout.json, draws debug overlays.
```

`atlas_ai/visible_extractor.py`

```text
Baseline visible extractor. Crops visible mockup regions into exact exported
BMP tensors, default-backing uncertain/hidden regions.
```

`atlas_ai/hidden_state_compiler.py`

```text
Deterministic plausible hidden states:
- VOLUME/BALANCE strips from visible track/thumb
- EQ slider frames by thumb translation
- CBUTTONS pressed states by shift/darken
- SHUFREP toggles by glow/dim
- PLAYPAUS/MONOSTER plausible alternates
```

`atlas_ai/torch_cranamp_renderer.py`

```text
Differentiable Cranamp-lite visible renderer for main/EQ/playlist product loss.
It is not full Cranamp. Use it for optimization/proxy diagnostics only; product
progress is judged on real Cranamp CLI renders.
```

`models/visible_skin_net.py`

```text
Initial V8 neural interface:
normalized mockup -> exact exported BMP tensors.
Hidden state synthesis remains downstream.
```

## Validation Already Run

Codex ran:

```bash
python -m py_compile \
  atlas_ai/v8_layout.py \
  atlas_ai/v8_assets.py \
  atlas_ai/visible_extractor.py \
  atlas_ai/hidden_state_compiler.py \
  atlas_ai/torch_cranamp_renderer.py \
  models/visible_skin_net.py \
  scripts/normalize_mockup.py \
  scripts/v8_mockup_to_skin.py \
  scripts/refine_to_mockup.py \
  scripts/eval_product_set.py
```

Codex ran:

```bash
python -m pytest \
  tests/test_v8_product_pipeline.py \
  tests/test_export_skin.py \
  tests/test_cranamp_cli.py
```

Result:

```text
12 passed
```

Codex smoke:

```bash
python scripts/v8_mockup_to_skin.py \
  --input assets/default_skin/MAIN.bmp \
  --out /tmp/uigen_v8_product_smoke \
  --default-skin assets/default_skin

cranamp_cli/cranamp-cli render-random \
  --skin-dir /tmp/uigen_v8_product_smoke/skin/skin.wsz \
  --seed 7 \
  --canvas-w 960 \
  --canvas-h 1728 \
  --out-view /tmp/uigen_v8_product_smoke/cranamp_render.png
```

The generated `.wsz` loaded through the existing Cranamp CLI renderer.

## Claude Run Ownership

Historical V8 run/eval sequence:

1. Create/populate `eval_mockups/` with 20-50 fixed generated mockups.
2. For each mockup, add manual rect JSON if default stacking is wrong.
3. Run `scripts/eval_product_set.py`.
4. Inspect `side_by_side.png` per case and fill human ratings in
   `summary.csv`.
5. Pick the biggest visible product failure and fix that first.

The current biggest failure is playlist/PLEDIT sprite reconstruction under real
Cranamp, measured with component metrics, not render-random state confound.

Do not optimize a subproblem unless it improves one of:

```text
loadability
visible render similarity
sharpness
component correctness
human rating
```

## Likely First Product Failures

Expected rough spots in the deterministic baseline:

```text
1. Manual layout is needed for non-stacked/generated compositions.
2. PLEDIT extraction is approximate; playlist chrome may be distorted.
3. Visible extractor is crop-based, not semantic; button and slider state
   guesses are crude.
4. Torch Cranamp-lite renderer is enough for optimization but not exact full
   Cranamp.
5. Hidden states are plausible, not learned.
```

Fix order should be:

```text
1. Better layout/normalization and manual correction workflow.
2. Better visible extractor/UV copy for visible components.
3. Better render-match refinement.
4. Better hidden-state compiler only if product screenshots show hidden states
   are visibly bad.
5. Train VisibleSkinNet only after product eval data and render gate are in place.
```

## Minimal Product Milestone

Target:

```text
10 generated mockups -> 10 skin.wsz files
```

Pass:

```text
all load in Cranamp
main/EQ/playlist visible style resembles input
buttons/sliders are recognizable and not mush
side-by-side review shows a usable V0
```

This is now the real gate. Hidden historical atlas MAE is not.
