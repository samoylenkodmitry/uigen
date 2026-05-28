# V9 RectifyNet Handoff

## Superseded By V10

This document is retained as history only.

As of 2026-05-28, the active direction is no longer V9 UV/copy/fallback
rectification. The current plan is V10:

```text
one full render/mockup image -> one expert model per output BMP -> final BMP pixels
```

See:

```text
HANDOFF_V10_BMP_EXPERTS.md
```

Do not continue V9 as the primary architecture unless V10 gates show a specific
reason to return to UV/copy supervision.

Date: 2026-05-28

## Why This Reset Exists

The current V8 screenshot shows the wrong architecture: crop-and-paste, not
learned fitting. The baseline takes visible rectangles from a mockup and writes
them into exported BMP slots. That can produce a loadable `.wsz`, but it cannot
solve the product problem because it does not semantically relocate or resize
elements into valid Winamp skin geometry.

Observed failure mode:

```text
main buttons leak into EQ
playlist content gets baked into playlist chrome
live main-window text/readout gets baked into MAIN.bmp
EQ sliders/buttons are copied as window pixels instead of compiled as sprites
```

Therefore:

```text
V8 crop extractor = loadability smoke test only
```

It is not product evidence and should not be optimized further as the primary
path.

The original product target remains:

```text
big-model-generated Winamp-like mockup
-> spatially grounded fidelity translation
-> exact Cranamp/Winamp skin files
-> real Cranamp render visually resembles the mockup
```

## V9 Goal

Implement:

```text
RectifyNetV9
```

Training task:

```text
distorted Cranamp render of a real skin
-> canonical exported BMP tensors from that same skin
```

The model must learn spatial fitting. It should learn that:

```text
transport row in the input -> CBUTTONS.bmp
EQ slider style -> EQMAIN / VOLUME / BALANCE structures
playlist chrome -> PLEDIT.bmp
playlist body text -> ignore or convert to colors/config
main display/readout -> ignore/procedural, not baked into MAIN.bmp
```

This is supervised inverse rendering from synthetic data, not rough manual atlas
slicing.

## Training Pair

For each real skin:

Target:

```text
exact exported BMP tensors from the original skin
```

Input:

```text
Cranamp render of that same skin, with randomized component-level distortions
```

The distorted input should look like a generated mockup error: same style and
rough Winamp composition, but wrong sizes, positions, spacing, and local
alignment.

Distortions to include:

```text
component dx / dy
component scale_x / scale_y
uniform scale
local component offsets
wrong button spacing
buttons crossing nearby regions
slider heads too large or too small
playlist scrollbar misplaced
titlebar height drift
EQ slider group shifted
transport row shifted/stretched
shuffle/repeat offset
volume/balance stretch
window spacing and crop drift
```

The target always remains the original exported skin assets.

## Data Generator Output

Build a V9 generator/dataset around the existing Cranamp renderer. Each sample
should write:

```text
input/distorted.png
target/files/*.bmp or target tensors
labels/provenance.npz
labels/component_rects.json
labels/state.json
debug/overlay.png
```

Use final-frame provenance as a teacher signal only. It is not available at
inference, but it gives perfect training supervision for visible
screen-to-asset correspondences.

Existing reusable code:

```text
atlas_ai/v6_labels.py
scripts/16_make_v6_dataset.py
models/slotnet_v6.py
```

`atlas_ai/v6_labels.py` already converts Cranamp final-frame provenance into:

```text
visible_mask  [H, W]
uv_target     [2, H, W] in grid_sample coordinates
```

`scripts/16_make_v6_dataset.py` already shows how to call:

```text
cranamp_cli.py rand_params(...)
cranamp_cli.py render_with_params(...)
renderer.provenance
```

V9 should extend this idea from source-preserving UV labels into the full
rectification task.

## RectifyNetV9 Architecture

Input:

```text
normalized distorted render / mockup
shape: [3, H, W]
initial target size: 960x1728 unless the current normalized size changes
```

Encoder:

```text
high-resolution multi-scale encoder
E1: stride 1 or 2 for texture/detail
E2: stride 4
E3: stride 8
E4: stride 16
```

Do not compress the image into a single global style vector. Prior work showed
that loses texture and encourages identity memorization.

Per-file heads output exact exported BMP tensors, not padded atlas images:

```text
MAIN.bmp
EQMAIN.bmp
PLEDIT.bmp
TITLEBAR.bmp
CBUTTONS.bmp
SHUFREP.bmp
POSBAR.bmp
VOLUME.bmp
BALANCE.bmp
MONOSTER.bmp
PLAYPAUS.bmp
```

For each output file pixel, predict:

```text
uv_grid       # where to sample from the input image
copy_conf     # whether visible input evidence should be copied
residual_rgb  # correction applied to copied color
fallback_rgb  # generated color if no visible evidence exists
```

Composition:

```python
copy_rgb = sample(input_image, uv_grid)
copy_rgb = copy_rgb + residual_rgb
final_rgb = copy_conf * copy_rgb + (1 - copy_conf) * fallback_rgb
```

This is the core difference from the crop baseline. The model predicts where to
sample from and how to canonicalize the result.

## Losses

Visible/copy losses, only where source asset pixels are visible in the distorted
render:

```text
L_uv        = predicted UV vs provenance UV
L_conf      = copy_conf vs visible mask
L_copy_rgb  = sampled/copied pixel vs target BMP pixel
```

Full exported-file losses, on supported file pixels:

```text
L_rgb
L_sobel
```

Do not let hidden-state pixels dominate early. The input cannot show every
hidden state, and the MVP should not be blocked on historical hidden-state
recovery.

Render loss:

```text
render(predicted skin, same visible state/layout)
vs
target clean render
```

For real GPT/Gemini mockups with no atlas target, later use:

```text
render(predicted skin) vs input mockup
```

as the product refinement objective.

## Hidden States

Keep hidden states deterministic for MVP.

Use code to synthesize plausible variants:

```text
visible slider style -> full VOLUME/BALANCE strips
visible EQ thumb style -> all EQ slider positions
normal button -> pressed button via shift/darken
toggle off -> toggle on via glow/dim
active titlebar -> inactive titlebar via desaturate/dim
```

Do not run hidden-state historical atlas recovery as the main project. Hidden
state models can return later as replacements for deterministic compiler
pieces, after visible rectification works.

## Gates

Use four gates. Do not launch 20 side experiments.

### Gate 1: One-Skin Rectification

```text
1 real skin
many distorted renders
target = exact exported BMP tensors
```

Pass:

```text
rendered predicted skin matches target clean render
visible files are sharp
exported visible-pixel MAE is low
buttons/sliders/playlist chrome land in the correct assets
```

### Gate 2: Seen-Style Rectification

```text
16 skins
train variants and validation variants from the same skins
```

Pass:

```text
model handles unseen distortions of known skins
```

### Gate 3: Held-Out Skin Rectification

```text
train skins
held-out skins
synthetic distortions
```

Pass:

```text
model fits components of new skins into valid assets
without just memorizing skin identity
```

### Gate 4: Product Mockups

```text
caat.png
goose.png
kittenamp.png
other generated mockups
```

Pass:

```text
skin.wsz loads
real Cranamp render visually resembles input
buttons/eq/playlist no longer leak into wrong regions
visible style and sharpness are preserved
```

If Gate 4 fails, fix visible rectification. Do not detour into hidden-state
completion unless the product render clearly shows hidden states are the
visible blocker.

## Stop As Primary Work

Do not spend run quota on:

```text
StateFamilyExpander S2/S3 variants
generic hidden completion experiments
V8 crop-baseline microfixes
proxy-only render metrics as product success
historical hidden atlas MAE as the main gate
```

The immediate next work is:

```text
RectifiedRenderDataset
V9 distorted-render generator with provenance labels
RectifyNetV9 UV/copy/residual/fallback heads
Gate 1 one-skin rectification
```
