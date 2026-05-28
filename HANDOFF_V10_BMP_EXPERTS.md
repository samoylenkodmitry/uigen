# V10 BMP Expert Pipeline Handoff

Date: 2026-05-28

## Reset

The active architecture is no longer V8 crop extraction or V9 UV/provenance
rectification. The clean target is:

```text
one generated Winamp-like full mockup image
-> one expert model per output BMP
-> exact final BMP pixels
-> collect BMPs + default runtime files
-> zip .wsz
-> render in Cranamp for human review
```

Stop these as primary work:

```text
primitive compiler
crop baseline
hidden-state side quests
rough inverse atlas
manual thumb placement
full-atlas model
StateFamilyExpander variants
```

The model family should directly learn:

```text
full generated/rendered skin image -> one exact BMP file
```

## Experts

Train separate models:

```text
Expert_MAIN:     input full render/mockup -> MAIN.bmp
Expert_EQMAIN:   input full render/mockup -> EQMAIN.bmp
Expert_PLEDIT:   input full render/mockup -> PLEDIT.bmp
Expert_TITLEBAR: input full render/mockup -> TITLEBAR.bmp
Expert_CBUTTONS: input full render/mockup -> CBUTTONS.bmp
Expert_VOLUME:   input full render/mockup -> VOLUME.bmp
Expert_BALANCE:  input full render/mockup -> BALANCE.bmp
Expert_POSBAR:   input full render/mockup -> POSBAR.bmp
Expert_SHUFREP:  input full render/mockup -> SHUFREP.bmp
Expert_MONOSTER: input full render/mockup -> MONOSTER.bmp
Expert_PLAYPAUS: input full render/mockup -> PLAYPAUS.bmp
```

`header.bmp` should be treated as:

```text
TITLEBAR.bmp
```

Each expert sees the entire normalized skin image, not a crop, and outputs one
BMP tensor at exact export size.

Use the current export spec as the source of truth for file names and sizes:

```text
atlas_ai/export_spec.py
```

Current trainable files:

```text
MAIN.bmp       275x115
TITLEBAR.bmp   344x87
CBUTTONS.bmp   136x36
SHUFREP.bmp     92x85
MONOSTER.bmp    56x24
PLAYPAUS.bmp    42x9
EQMAIN.bmp     275x315
PLEDIT.bmp     280x186
POSBAR.bmp     307x10
VOLUME.bmp      68x433
BALANCE.bmp     47x433
```

## Training Data

For every source skin:

```text
target:
  exact BMP file from the original/normalized skin

input:
  Cranamp full skin render with randomized state and randomized component geometry
```

For one skin:

```text
render_000.png -> MAIN.bmp
render_000.png -> VOLUME.bmp
render_000.png -> CBUTTONS.bmp

render_001.png -> MAIN.bmp
render_001.png -> VOLUME.bmp
render_001.png -> CBUTTONS.bmp

...
```

The target BMP stays the same for that skin. The input render changes across
states and transforms.

This is how hidden states become learnable without a primitive compiler:

```text
Expert_VOLUME sees many renders of the same skin:
  volume at position 0
  volume at position 1
  volume at position 2
  ...
target is always full VOLUME.bmp

Therefore it learns:
  visible volume state + whole skin style -> full VOLUME.bmp
```

The same applies to EQ, balance, buttons, play/pause, shuffle/repeat, and
playlist.

## Cranamp Data Generation

For each skin, generate a state-covered dataset. Do not rely on infinite random
mixtures alone. Generate complete coverage for target families plus random
global mixtures.

Concrete driver:

```text
params = cranamp_cli.py rand_params(seed, 960, 1728)
override params["state"] for the explicit state being covered
keep params["component_transforms"] and params["window_scales"] randomized
render = cranamp_cli.py render_with_params(skin_source, params, 960, 1728)
save renderer.canvas as renders/<skin>_<variant>.png
save params JSON as states/<skin>_<variant>.json
```

Do not use plain `render-random` as the V10 generator. It is fine as a helper
concept, but V10 needs deterministic coverage: every volume position, every
balance position, button states, EQ states, and so on must appear by
construction.

Required state coverage per skin:

```text
VOLUME.bmp:
  render all volume positions at least once

BALANCE.bmp:
  render all balance positions at least once

CBUTTONS.bmp:
  render no pressed button
  render prev/play/pause/stop/next/eject pressed

SHUFREP.bmp:
  render all shuffle/repeat/eq/pl toggle combinations needed by Cranamp

MONOSTER.bmp:
  render mono/stereo states

PLAYPAUS.bmp:
  render play/pause/stop states

POSBAR.bmp:
  render many seek positions, including min/max/middle

EQMAIN.bmp:
  render each EQ slider band at all or most legal positions
  render random EQ curves
  render EQ on/off
  render EQ auto on/off

PLEDIT.bmp:
  render different scroll positions
  render different selected rows
  render different playlist heights if supported
  render different bottom button states

MAIN.bmp and TITLEBAR.bmp:
  render different playback states
  render titlebar active/inactive if applicable
  render different visualizer/time/text states
  render random component offsets/scales
```

First full coverage schedule per skin:

```text
base neutral/random geometry:
  32 variants

VOLUME:
  28 variants, volume = i / 27

BALANCE:
  28 variants, balance = i / 27

CBUTTONS:
  7 variants, pressed_transport_button = -1,0,1,2,3,4,5

SHUFREP:
  16 variants, all shuffle/repeat/eq_on/pl_toggle combinations that Cranamp
  exposes in params/state; if pl_toggle is not render-state controlled yet,
  still sweep shuffle/repeat/eq_on and record the gap.

MONOSTER:
  2 variants if mono/stereo state is exposed; otherwise record as renderer gap.

PLAYPAUS:
  3 variants, playback = playing/paused/stopped

POSBAR:
  29 variants, posbar = i / 28

EQMAIN:
  308 single-band variants, for band j in 0..10 and position i in 0..27
  96 random-curve variants
  4 eq_on/eq_auto combinations

PLEDIT:
  64 variants over playlist_scroll and playlist_selected_row

extra random mixtures:
  128 variants
```

This is roughly 700-750 variants per skin. For local smoke, implement the same
variant taxonomy but cap each family hard (for example 2-4 samples per family)
so tests run quickly.

Every render should also include generated-mockup-like geometry distortions:

```text
component dx/dy
scale_x
scale_y
uniform scale
wrong button spacing
wrong slider size
wrong slider position
titlebar drift
EQ slider group shift
playlist scrollbar shift
transport row shift
```

The input is wrong-looking. The target is always the correct original BMP.

## Concrete Data/Run Choices

Use these choices unless there is a concrete blocker.

Skin sources:

```text
Gate 1 plumbing smoke:
  assets/default_skin

Gate 1 real one-skin:
  data_v7_16skin_completion/minimalistic_black_145917e6

Gate 1 hard sanity after the real one-skin pass:
  data_v7_16skin_completion/the_four_horsemen_523e6bdf
  data_v7_16skin_completion/goodgawd_bba84deb

Gate 2:
  all 14 existing dirs in data_v7_16skin_completion
```

`data_v7_16skin_completion` currently has 14 unpacked skin dirs, not 16. Do not
pretend otherwise. Run this as `Gate 2-14` first. After Gate 1 works, add two
more valid skins from `skins_raw/` using the existing unpack/export tooling and
then repeat as the true 16-skin gate.

Variant counts:

```text
local smoke/plumbing: 16-32 variants
Gate 1 one-skin: 512-768 variants
Gate 2-14: 384-768 variants per skin
```

Kaggle/long-run rule:

```text
local:
  code/tests
  dataset smoke
  forward pass
  tiny MAIN.bmp overfit

Kaggle or long-run machine:
  any full expert training expected to exceed a few minutes
  Gate 1 for all 11 experts
  Gate 2 and above
```

Training order:

```text
1. MAIN.bmp expert first, because it is the broadest visible sanity check.
2. CBUTTONS.bmp next, because it catches semantic fitting of transport buttons.
3. EQMAIN.bmp next, because it tests state-rich output from partial visual cues.
4. PLEDIT.bmp next, because it tests chrome vs dynamic text.
5. Then the remaining smaller state files.
```

Do not launch 11 jobs until MAIN proves the dataset/model/training loop can
overfit the one-skin gate.

## Dataset Layout

Use a simple layout:

```text
data_v10/
  renders/
    skin_000001_000.png
    skin_000001_001.png
    ...

  states/
    skin_000001_000.json
    skin_000001_001.json
    ...

  targets/
    skin_000001/
      MAIN.bmp
      EQMAIN.bmp
      PLEDIT.bmp
      TITLEBAR.bmp
      CBUTTONS.bmp
      VOLUME.bmp
      BALANCE.bmp
      POSBAR.bmp
      SHUFREP.bmp
      MONOSTER.bmp
      PLAYPAUS.bmp

  csv/
    train_MAIN.csv
    train_EQMAIN.csv
    train_PLEDIT.csv
    train_TITLEBAR.csv
    train_CBUTTONS.csv
    train_VOLUME.csv
    train_BALANCE.csv
    train_POSBAR.csv
    train_SHUFREP.csv
    train_MONOSTER.csv
    train_PLAYPAUS.csv
```

Each CSV row:

```text
render_png,target_bmp,skin_id,variant_id,state_json
```

Example:

```text
renders/skin_000001_023.png,targets/skin_000001/VOLUME.bmp,skin_000001,023,states/skin_000001_023.json
```

No atlas target. No primitive target. No per-pixel visible mask required for the
V10 baseline.

## BMPExpertNet

Use one architecture template for all BMP experts.

Input:

```text
full normalized Cranamp/mockup render
shape: [3, 1728, 960]
```

Output:

```text
exact BMP tensor [3, H, W]
```

Starting architecture:

```text
Encoder:
  ConvNeXt/ResNet-style CNN with FPN
  E1 stride 2
  E2 stride 4
  E3 stride 8
  E4 stride 16

Features:
  multi-scale feature maps, not only global style

Decoder:
  target BMP query grid at H/4 x W/4
  Fourier x/y coordinate channels
  output-query cross-attention into encoder features
  2 cross-attention layers
  4 heads
  nearest-upsample residual decoder
  final RGB head
```

Concrete starting config:

```text
encoder base channels = 48
attention dim = 256
decoder channels = 128
query grid = target H/4 x target W/4
attention heads = 4
cross-attention layers = 2
```

Do not use bilinear upsampling in the final decoder. Use nearest upsample plus
residual conv blocks to avoid soft output.

Output heads:

```text
rgb_logits [3,H,W]
special_logits [K,H,W] optional later
```

V10 baseline trains RGB first. Add exact special-color snapping or CE later only
if #FF00FF/key-color behavior is a measured blocker.

## Loss

Train on actual BMP pixels only:

```text
L = 1.0 * L1_RGB
  + 1.5 * Sobel_RGB
  + 0.5 * Laplacian_RGB
```

Metrics:

```text
MAE
hit_5_255
Sobel MAE
rendered skin side-by-side after zipping all experts
```

Do not optimize padded atlas pixels. Do not average across unused regions.

## Training Gates

Do not train all experts at full scale first.

### Gate 1: One-Skin Overfit Per Expert

Pick one skin.

For each expert:

```text
input = all generated renders of that one skin
target = that skin's BMP
```

Pass:

```text
MAE < 0.01
hit_5_255 > 0.90
visual BMP sharp
```

This must pass.

### Gate 2: 16-Skin Memorization

For each expert:

```text
16 skins
many variants per skin
train and evaluate same skins
```

Pass:

```text
retrieval/top-style behavior correct by visual inspection
median MAE < 0.02
hit_5_255 > 0.85
```

### Gate 3: Held-Out Skin Generalization

For each expert:

```text
train skins
held-out skins
```

Pass:

```text
output resembles held-out target BMP
not average/default
```

### Gate 4: Product Mockups

Run:

```text
caat.png
goose.png
kittenamp.png
...
```

Pipeline:

```text
input mockup
-> all BMP experts
-> zip script
-> Cranamp render
-> human review
```

Pass:

```text
skin.wsz loads
buttons are in MAIN/CBUTTONS correctly
EQ does not receive main buttons
playlist content is not blindly baked into chrome
output resembles input
```

## Inference

Implement:

```bash
python infer_v10.py --image caat.png --out out_skin/
```

Internally:

```text
load input image
normalize to training render size
run Expert_MAIN
run Expert_EQMAIN
run Expert_PLEDIT
run Expert_TITLEBAR
run Expert_CBUTTONS
run Expert_VOLUME
run Expert_BALANCE
run Expert_POSBAR
run Expert_SHUFREP
run Expert_MONOSTER
run Expert_PLAYPAUS
copy/default TEXT, NUMBERS, configs
zip .wsz
render with Cranamp
save side-by-side
```

Use existing packaging helpers where possible:

```text
atlas_ai/v8_assets.py save_exported_tensors(...)
atlas_ai/v8_assets.py package_skin_dir(...)
```

These helpers can stay named `v8_assets.py` for now; do not spend time on a
rename unless it becomes confusing in code.

## Implementation Tasks

Implement:

```text
scripts/make_v10_bmp_expert_dataset.py
atlas_ai/dataset_v10_bmp.py
models/bmp_expert_net.py
train_bmp_expert.py
scripts/eval_bmp_expert.py
infer_v10.py
tests/test_v10_bmp_dataset.py
tests/test_bmp_expert_net.py
tests/test_infer_v10_packaging.py
```

Minimum first deliverable:

```text
1. dataset generator creates per-BMP CSVs for one skin
2. dataset loads render_png + target_bmp for one requested BMP
3. BMPExpertNet forward pass returns exact [B,3,H,W]
4. train_bmp_expert.py can overfit a tiny one-skin MAIN.bmp smoke sample
5. infer_v10.py packages predicted/default BMPs into a loadable skin.wsz
```

## What Counts As Success

Product mockups are judged by rendered output:

```text
skin.wsz loads
main/EQ/playlist render matches input composition
buttons are in button assets, not leaking into EQ
EQ assets do not receive main-window button crops
playlist chrome is separated from playlist text
mascot/decorative art is preserved where format allows
output is sharp enough
```

Do not report "it loads" or "MAE moved" as success if screenshots still show
semantic leakage.
