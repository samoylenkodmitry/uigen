# Cranamp Render-to-Atlas Inverter V0 Roadmap — v4

- [x] Treat this file as the single implementation roadmap for the coding agent.
- [x] Keep this roadmap in the fresh repo as `roadmap.md`.
- [x] The user will handle phase-zero skin collection separately.
- [x] The coding agent will do all implementation work in one repo; do not split into separate agents.
- [ ] V0 goal: learn `randomized Cranamp render of a known skin -> exact packed atlas of that same skin`.
- [ ] V0 inference path: `AI-generated Winamp-like mockup -> GeoNet80 -> SlotNetV1 -> packed atlas -> split atlas into BMP files -> .wsz/Cranamp skin folder`.
- [ ] V0 is a foundation model, not the final creative KITTENAMP/GPT/Gemini product.
- [ ] Do not judge V0 by creative AI mockups; use creative AI mockups only for qualitative external tests after V0 works.
- [ ] Do not implement a rough inverse atlas.
- [ ] Do not implement fake atlas labels from generated mockups.
- [ ] Do not manually slice AI-generated mockups into target atlases.
- [ ] Do not use creative AI restyles as paired supervised data against the original atlas.
- [ ] Do not implement custom Cranamp skin format in V0.
- [ ] Do not implement per-slider custom sprites in V0.
- [ ] Do not implement diffusion/img2img dataset generation in V0.
- [ ] Do not implement ControlNet/LoRA/GAN/palette-index output in V0.
- [ ] Do not implement a differentiable compositor in V0.
- [ ] Do not implement unpaired creative render-loss optimization in V0.
- [ ] Do not implement an atlas autoencoder prior in V0.
- [ ] Prepare for future extension only by reserving rect classes, state slots, and atlas slots.


## Revision notes from Gemini/Opus review

- [x] Keep the offline saved dataset pipeline; do not switch V0 to on-the-fly/FFI data generation.
- [ ] Add Cranamp render-throughput benchmarking before generating large datasets.
- [ ] Add explicit Cranamp work for `visible_atlas_mask`; this is renderer instrumentation, not a trivial script.
- [ ] Verify Cranamp magenta behavior before enabling magenta snapping/loss.
- [ ] Replace GeoNet global-average state regression with local component-anchored state regression.
- [ ] Use EQ slider group as the primary detected geometry; derive individual EQ band rects from the group in V0.
- [ ] Add random letterbox/padding augmentation for GeoNet and SlotNet.
- [ ] Add conditional shallow SlotNet path for tiny slots.
- [ ] Keep RGB output in V0, but strengthen edge loss and add a blur-control escalation path.
- [x] Make export-size validity a 100% hard gate.
- [ ] Make `--rect-mix-ratio 0.5` explicit for SlotNet Stage B.
- [ ] Add state-anchor jitter for GeoNet state regression to avoid train/inference brittleness.
- [ ] Specify GeoNet state feature sampling as 3x3 average pooling on FPN `P`, not single-pixel lookup.
- [ ] Make inference state decoding explicitly two-pass: detect rects first, then sample predicted anchors and decode states.
- [ ] Add SlotNet scale-ratio conditioning channels so the model knows how much the source crop was down/up-sampled.
- [ ] Track rare pressed-state/hidden-state slot metrics; only increase variants or enable state-balanced rendering if metrics show muddy state sprites.
- [ ] Specify visible-atlas provenance as a separate `u32` ID buffer to avoid destroying Cranamp RGB render hot-path throughput.
- [x] Use a per-slot magenta policy schema, not a single global magenta boolean. (Verified against 100-skin sample: 3-7% of skins per slot use #FF00FF; Cranamp keys all magenta to alpha=0 globally; policy `default: true, per_slot: all true` is correct.)
- [x] Add deterministic EQ band subdivision unit test shared by training and inference.
- [ ] Add gradient accumulation fallback for GeoNet/SlotNet OOM instead of silently reducing real batch too far.
- [ ] Clarify Smoke training validates pipeline/loadability only, not visual quality.
- [ ] Keep differentiable compositor, atlas prior, creative render-loss, and custom Cranamp extensions out of V0.

## Revision notes from final Gemini/Opus review

- [ ] Keep V0 architecture unchanged; only add implementation hardening items from final review.
- [x] Use mask-sum-normalized RGB/Sobel losses, not flat tensor means, so sparse/padded slots train at the same active-pixel gradient scale as dense slots.
- [x] Add `configs/state_regions_v1.json` so pressed/normal/hidden-state metrics are not guessed by the agent.
- [x] Expose `--state-balanced` in the Cranamp CLI and dataset script, but keep it disabled by default.
- [x] Change SlotNet scale conditioning from raw ratios to log-ratios: `log_scale_x = log(max(scale_x, 1e-6))`, `log_scale_y = log(max(scale_y, 1e-6))`.
- [x] Cap GeoNet state-anchor jitter by component size: no more than ±2 stride-4 cells and no more than ±25% of the anchor width/height in grid cells.
- [ ] Add a numeric rare-state failure trigger: if `pressed_state_rgb_mae > 1.5 * normal_state_rgb_mae` for any state-bearing slot, regenerate with `--variants 32 --state-balanced`.
- [x] Keep offline saved datasets; do not adopt on-the-fly FFI generation for V0.
- [x] Keep VGG/perceptual/GAN loss out of default V0.
- [ ] Treat v4 as the final handoff spec unless implementation discovers a concrete Cranamp/runtime bug.

## 0. Hard project framing

- [x] Source training truth is always a real existing skin folder or `.wsz`.
- [ ] Each source skin contains separate BMP files, for example `MAIN.bmp`, `EQMAIN.bmp`, `PLEDIT.bmp`, `CBUTTONS.bmp`, `TITLEBAR.bmp`, `VOLUME.bmp`, `BALANCE.bmp`, `SHUFREP.bmp`, `POSBAR.bmp`, `MONOSTER.bmp`, `PLAYPAUS.bmp`, `TEXT.bmp`, `NUMBERS.bmp`, and optional extras.
- [x] Pack each source skin’s BMP files into one fixed `1024x1024` atlas; that packed atlas is the model target.
- [ ] Modify or fork Cranamp so it can render the source skin into randomized 3-window views and also output exact component rects/states.
- [ ] Generate training data forward only: `source skin -> packed atlas target` and `source skin -> randomized Cranamp render input + rect/state labels`.
- [x] Save generated views/rects/states/masks to disk before training; do not generate Cranamp renders inside the PyTorch dataloader in V0.
- [ ] If disk I/O becomes a bottleneck, pack offline samples into tar/LMDB-style shards; do not switch the V0 design to on-the-fly rendering.
- [ ] Model1 `GeoNet80` learns `view.png -> rect vector + state vector`.
- [ ] Model2 `SlotNetV1` learns `view.png + rect vector + state vector -> source skin atlas slots`.
- [ ] Tool inference uses `mockup.png -> GeoNet80 -> SlotNetV1 -> atlas -> exported BMP files -> skin.wsz`.
- [ ] Use strict classic Winamp/Cranamp skin outputs for V0.
- [ ] Use RGB output for V0.
- [ ] Keep technical/special output channels in SlotNet from day one, but only train useful special classes initially.
- [ ] Exclude `TEXT.bmp` and `NUMBERS.bmp` from V0 loss because Cranamp will use high-resolution/system fonts.
- [ ] Treat creative AI mockups as a V1/V2/V3 problem requiring render-similarity/refinement or Cranamp extension slots.

## 1. Repository structure

- [ ] Create repo root `cranamp-atlas-ai/`.
- [x] Create `roadmap.mf` at repo root.
- [x] Create `requirements.txt`.
- [x] Create `configs/`.
- [x] Create `configs/atlas_v1.json`.
- [x] Create `configs/components_v1.json`.
- [x] Create `configs/slot_sources_v1.json`.
- [x] Create `configs/state_regions_v1.json`.
- [x] Create `configs/export_profile_classic.json`.
- [x] Create `configs/train_v0.yaml`.
- [x] Create `assets/default_skin/` for fallback/default classic assets.
- [x] Create `cranamp_cli/` for the Cranamp fork/submodule/wrapper.
- [x] Create `cranamp_cli/README.md` documenting how to point to the user’s Cranamp repo.
- [x] Create `scripts/00_scan_skins.py`.
- [x] Create `scripts/01_pack_skins.py`.
- [x] Create `scripts/02_render_dataset.py`.
- [x] Create `scripts/03_make_splits.py`.
- [x] Create `scripts/04_check_dataset.py`.
- [x] Create `scripts/05_export_atlas_to_skin.py`.
- [x] Create `scripts/06_benchmark_cranamp.py`.
- [x] Create `scripts/07_verify_magenta.py`.
- [x] Create `scripts/08_pack_dataset_shards.py` as optional fallback if small-file I/O becomes a bottleneck.
- [x] Create `models/geonet80.py`.
- [x] Create `models/slotnet_v1.py`.
- [x] Create `models/losses.py`.
- [x] Create `models/crop.py`.
- [x] Create `models/atlas.py`.
- [x] Create `train_geonet.py`.
- [x] Create `train_slotnet.py`.
- [x] Create `eval_pipeline.py`.
- [x] Create `infer_skin.py`.
- [x] Create `tests/test_atlas_pack.py`.
- [x] Create `tests/test_rect_encoding.py`.
- [x] Create `tests/test_slot_crop.py`.
- [x] Create `tests/test_export_skin.py`.
- [x] Create `debug/` for generated debug visualizations, not committed.
- [x] Create `data/` for generated datasets, not committed.
- [x] Add `.gitignore` entries for `data/`, `debug/`, `runs/`, `eval/`, `*.wsz`, and large generated files.

## 2. Python/runtime stack

- [ ] Use Python 3.11.
- [ ] Use PyTorch 2.x.
- [ ] Use torchvision.
- [ ] Use NumPy.
- [ ] Use Pillow.
- [ ] Use OpenCV Python.
- [ ] Use pandas.
- [ ] Use tqdm.
- [ ] Use PyYAML.
- [ ] Use matplotlib for debug grids.
- [ ] Use TensorBoard for logging.
- [ ] Use safetensors for model checkpoints.
- [ ] Do not require HuggingFace/diffusers for V0.
- [ ] Do not require wandb; optional only.

`requirements.txt` must include:

```text
torch
torchvision
numpy
pillow
opencv-python
pandas
tqdm
pyyaml
matplotlib
tensorboard
safetensors
```

## 3. Phase zero skin collection input contract

- [ ] The user will download/collect as many Winamp skins as possible.
- [x] The repo must expect raw skins at `skins_raw/`.
- [x] Support `.wsz` zip files under `skins_raw/`.
- [x] Support plain unpacked skin directories under `skins_raw/`.
- [x] Support skin directories containing mixed-case filenames.
- [x] Normalize filenames case-insensitively internally, e.g. `MAIN.bmp`, `main.bmp`, `Main.BMP` all map to `MAIN.bmp`.
- [x] Preserve original file paths in metadata for debugging.
- [x] Do not require every optional BMP to exist.
- [ ] Reject only skins Cranamp cannot load/render at all.

Expected raw input examples:

```text
skins_raw/
  skin_000001.wsz
  skin_000002/
    MAIN.bmp
    EQMAIN.bmp
    PLEDIT.bmp
    CBUTTONS.bmp
    TITLEBAR.bmp
    VOLUME.bmp
    BALANCE.bmp
    SHUFREP.bmp
    POSBAR.bmp
    MONOSTER.bmp
    PLAYPAUS.bmp
    TEXT.bmp
    NUMBERS.bmp
  skin_000003.wsz
```

## 4. Cranamp fork/CLI

- [x] Fork or wrap the user’s Cranamp repo.
- [ ] If the repo path is not embedded, read it from environment variable `CRANAMP_REPO=/path/to/cranamp`.
- [x] Inspect Cranamp’s actual skin loader, renderer constants, fallback assets, and export dimensions.
- [ ] Do not guess exact BMP export dimensions from external sources.
- [ ] Do not hardcode uncertain classic dimensions if Cranamp already defines them.
- [x] Add or expose a CLI executable named `cranamp-cli`.
- [x] Implement `cranamp-cli dump-classic-spec`.
- [x] Implement `cranamp-cli render-random`.
- [x] Implement `cranamp-cli render-with-params`.
- [x] Make all Cranamp randomization deterministic from seed.
- [x] Make the Cranamp CLI output exact rect labels in the normalized format defined below.
- [x] Make the Cranamp CLI output exact state labels in the fixed vector format defined below.
- [x] Make the Cranamp CLI output a visible-atlas mask for each render.
- [x] Make the Cranamp CLI output replayable `params.json` for each render.
- [x] Make `render-with-params` replay the exact render from `params.json` for validation/debug.
- [x] Add `--state-balanced true|false` to `render-random`; default is `false`.
- [x] In `--state-balanced true`, override normal Bernoulli/random state sampling to enumerate pressed and toggle states systematically across variants.
- [x] Keep `--state-balanced` deterministic from seed and variant index.

Implementation note: the first `cranamp-cli` is a deterministic compositor in the vendored Cranamp fork, using Cranamp sprite constants and skin decode behavior. It satisfies the dataset file contract now; direct instrumentation of the interactive Cranamp renderer and exact UI parity remain open below.

Required CLI commands:

```bash
cranamp-cli dump-classic-spec \
  --out configs/export_profile_classic.json
```

```bash
cranamp-cli render-random \
  --skin-dir <skin_dir> \
  --seed <int> \
  --canvas-w 960 \
  --canvas-h 1728 \
  --out-view <png> \
  --out-rects <rects.f32> \
  --out-state <state.f32> \
  --out-visible-atlas-mask <mask.png> \
  --out-params <params.json> \
  --state-balanced false
```

```bash
cranamp-cli render-with-params \
  --skin-dir <skin_dir> \
  --params <params.json> \
  --canvas-w 960 \
  --canvas-h 1728 \
  --out-view <png>
```

Cranamp renderer instrumentation tasks:

- [ ] Add atlas-provenance tracking to Cranamp blit/compositing code.
- [ ] Keep the normal RGB compositing hot path and atlas provenance tracking logically separate; do not interleave provenance writes into inner RGB blend loops if that hurts throughput.
- [ ] Render or accumulate an auxiliary `u32` atlas-provenance ID buffer alongside the RGB output.
- [ ] Use `0` in the provenance buffer for background/no-atlas-source pixels.
- [ ] Encode drawn atlas pixels as `id = 1 + (slot_id << 20) + (atlas_y << 10) + atlas_x` because `atlas_x` and `atlas_y` are each `< 1024`.
- [ ] Decode provenance IDs at the end of the render to set `visible_atlas_mask[atlas_y, atlas_x] = 255` for every nonzero ID.
- [ ] For every source sprite pixel drawn to the output render, mark the corresponding packed-atlas pixel in `visible_atlas_mask`.
- [ ] Track provenance through scaled draws, local component shifts, pressed-state sprite selection, slider frame selection, and playlist/EQ/main window drawing.
- [ ] Do not approximate `visible_atlas_mask` with a whole-slot mask unless an explicit temporary fallback flag `--visible-mask-fallback whole-slot` is used for smoke testing.
- [ ] Add a debug skin whose atlas pixels encode unique IDs/colors, render it, and verify `visible_atlas_mask` marks exactly the drawn source regions.
- [x] Add `scripts/06_benchmark_cranamp.py` to time `render-random` throughput for 100, 1000, and 8000 renders.
- [x] Benchmark cold-start throughput and warmed parallel-process throughput separately. (Sequential measured: ~2.0 renders/s = 0.50 s/render; bottlenecked by Python subprocess startup, not render work. 32 cores available.)
- [ ] If average render time exceeds `0.50 s/render`, make `scripts/02_render_dataset.py` run multiple Cranamp processes in parallel. (At threshold — parallelization needed for V0/Full datasets.)
- [ ] Log renders/second, failures, CPU utilization, and disk write throughput before generating V0/V1 datasets.

## 5. Fixed image sizes and coordinate conventions

- [x] Use fixed input render width `INPUT_W = 960` (both dims divisible by 32 for CNN).
- [x] Use fixed input render height `INPUT_H = 1728` (both dims divisible by 32 for CNN).
- [x] Input tensor shape is `[3, 1728, 960]`.
- [x] Normalize x coordinates by `960`.
- [x] Normalize y coordinates by `1728`.
- [ ] Use fixed target atlas width `ATLAS_W = 1024`.
- [ ] Use fixed target atlas height `ATLAS_H = 1024`.
- [ ] Target atlas tensor shape is `[3, 1024, 1024]`.
- [ ] Target atlas PNG mode is RGB.
- [ ] V0 target atlas has no alpha channel.
- [ ] Use letterboxing for arbitrary inference images into `960x1728`.
- [ ] Preserve letterbox transform metadata for potential future unletterboxing/debug.
- [ ] Training augmentation must include random letterbox/pad simulation so GeoNet sees inputs similar to arbitrary AI mockup aspect ratios.

## 6. Atlas V1 layout

- [ ] Create a fixed packed atlas named `ATLAS_V1_RGB`.
- [ ] Use `1024x1024x3` RGB atlas.
- [ ] Capacity rectangles are larger than some exact BMP dimensions; exact export dimensions come from `export_profile_classic.json`.
- [ ] Paste each source BMP into its slot top-left.
- [ ] Keep slot padding ignored in loss via the atlas mask.
- [ ] V0 active prediction slots are `MAIN`, `TITLEBAR`, `CBUTTONS`, `SHUFREP`, `MONOSTER`, `PLAYPAUS`, `EQMAIN`, `PLEDIT`, `POSBAR`, `VOLUME`, `BALANCE`.
- [ ] V0 ignored/model-output-but-no-loss slots are `NUMBERS`, `TEXT`, `EQ_EX`, `GEN`, `VIDEO`, `RESERVED_A`, `RESERVED_B`.
- [ ] Do not train `TEXT` or `NUMBERS` in V0.
- [ ] Do not train `GEN`, `VIDEO`, or extension slots in V0.
- [ ] Keep `RESERVED_A` and `RESERVED_B` for future extension without changing the atlas canvas.

Create `configs/atlas_v1.json` with this exact content:

```json
{
  "canvas_w": 1024,
  "canvas_h": 1024,
  "slots": [
    {"id": 0,  "name": "MAIN",       "file": "MAIN.bmp",     "x": 0,   "y": 0,   "w": 320, "h": 128, "loss_weight": 1.0},
    {"id": 1,  "name": "TITLEBAR",   "file": "TITLEBAR.bmp", "x": 320, "y": 0,   "w": 320, "h": 96,  "loss_weight": 2.0},
    {"id": 2,  "name": "CBUTTONS",   "file": "CBUTTONS.bmp", "x": 640, "y": 0,   "w": 160, "h": 64,  "loss_weight": 4.0},
    {"id": 3,  "name": "SHUFREP",    "file": "SHUFREP.bmp",  "x": 800, "y": 0,   "w": 128, "h": 128, "loss_weight": 4.0},
    {"id": 4,  "name": "MONOSTER",   "file": "MONOSTER.bmp", "x": 928, "y": 0,   "w": 64,  "h": 32,  "loss_weight": 4.0},
    {"id": 5,  "name": "PLAYPAUS",   "file": "PLAYPAUS.bmp", "x": 928, "y": 32,  "w": 64,  "h": 32,  "loss_weight": 4.0},
    {"id": 6,  "name": "NUMBERS",    "file": "NUMBERS.bmp",  "x": 640, "y": 64,  "w": 160, "h": 32,  "loss_weight": 0.0},
    {"id": 7,  "name": "TEXT",       "file": "TEXT.bmp",     "x": 320, "y": 96,  "w": 192, "h": 32,  "loss_weight": 0.0},
    {"id": 8,  "name": "EQMAIN",     "file": "EQMAIN.bmp",   "x": 0,   "y": 128, "w": 320, "h": 384, "loss_weight": 1.0},
    {"id": 9,  "name": "PLEDIT",     "file": "PLEDIT.bmp",   "x": 320, "y": 128, "w": 320, "h": 256, "loss_weight": 1.0},
    {"id": 10, "name": "POSBAR",     "file": "POSBAR.bmp",   "x": 320, "y": 384, "w": 320, "h": 32,  "loss_weight": 4.0},
    {"id": 11, "name": "EQ_EX",      "file": "EQ_EX.bmp",    "x": 320, "y": 416, "w": 320, "h": 64,  "loss_weight": 0.0},
    {"id": 12, "name": "VOLUME",     "file": "VOLUME.bmp",   "x": 640, "y": 128, "w": 96,  "h": 448, "loss_weight": 3.0},
    {"id": 13, "name": "BALANCE",    "file": "BALANCE.bmp",  "x": 736, "y": 128, "w": 80,  "h": 448, "loss_weight": 3.0},
    {"id": 14, "name": "GEN",        "file": "GEN.bmp",      "x": 0,   "y": 512, "w": 320, "h": 320, "loss_weight": 0.0},
    {"id": 15, "name": "VIDEO",      "file": "VIDEO.bmp",    "x": 320, "y": 512, "w": 320, "h": 320, "loss_weight": 0.0},
    {"id": 16, "name": "RESERVED_A", "file": null,           "x": 640, "y": 576, "w": 384, "h": 208, "loss_weight": 0.0},
    {"id": 17, "name": "RESERVED_B", "file": null,           "x": 640, "y": 800, "w": 384, "h": 224, "loss_weight": 0.0}
  ]
}
```

## 6A. State-region metric config

- [ ] Create `configs/state_regions_v1.json` before implementing pressed/hidden-state metrics.
- [ ] Do not let the coding agent guess which pixels are normal/pressed/hidden inside sprite-strip slots.
- [ ] Define state-bearing subregions in atlas-slot-local coordinates, not global atlas coordinates.
- [ ] Use `state_regions_v1.json` only for metrics/debug/escalation in V0, not for training supervision unless explicitly needed later.
- [ ] Include at least these state-bearing slots: `CBUTTONS`, `SHUFREP`, `PLAYPAUS`, `VOLUME`, `BALANCE`, and `POSBAR`.
- [ ] For each region, include `slot`, `name`, `state_type`, `x`, `y`, `w`, `h`, and optional `notes`.
- [ ] Agent must fill final region coordinates by inspecting Cranamp’s atlas interpretation and sprite-strip layout.
- [ ] Add unit test `tests/test_state_regions.py` verifying every region lies inside its slot capacity.
- [ ] Add unit test `tests/test_state_regions.py` verifying named state regions do not overlap accidentally unless `allow_overlap=true` is set.

Expected schema for `configs/state_regions_v1.json`:

```json
{
  "regions": [
    {"slot": "CBUTTONS", "name": "transport_normal", "state_type": "normal", "x": 0, "y": 0, "w": 136, "h": 18, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "CBUTTONS", "name": "transport_pressed", "state_type": "pressed", "x": 0, "y": 18, "w": 136, "h": 18, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "SHUFREP", "name": "shuffle_repeat_normal", "state_type": "normal", "x": 0, "y": 0, "w": 128, "h": 64, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "SHUFREP", "name": "shuffle_repeat_pressed", "state_type": "pressed", "x": 0, "y": 64, "w": 128, "h": 64, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "PLAYPAUS", "name": "playpaus_visible_states", "state_type": "state", "x": 0, "y": 0, "w": 64, "h": 32, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "VOLUME", "name": "volume_slider_strip", "state_type": "slider_strip", "x": 0, "y": 0, "w": 96, "h": 448, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "BALANCE", "name": "balance_slider_strip", "state_type": "slider_strip", "x": 0, "y": 0, "w": 80, "h": 448, "notes": "placeholder; fill from Cranamp/spec"},
    {"slot": "POSBAR", "name": "posbar_states", "state_type": "slider_strip", "x": 0, "y": 0, "w": 320, "h": 32, "notes": "placeholder; fill from Cranamp/spec"}
  ]
}
```

## 7. Export profile

- [ ] Generate `configs/export_profile_classic.json` by inspecting Cranamp.
- [ ] Do not rely on placeholder dimensions in this roadmap as final truth.
- [ ] Use the export profile during inference/export to crop each atlas slot to exact BMP dimensions.
- [ ] At inference, there is no target-skin `meta.json`; therefore export must use `export_profile_classic.json`.
- [ ] For each output BMP, crop from the top-left of that slot using export profile width/height.
- [ ] For `TEXT.bmp` and `NUMBERS.bmp`, copy default assets or generate blank/default assets accepted by Cranamp.
- [ ] For `PLEDIT.TXT`, use default file or derive simple colors from predicted atlas.
- [ ] For `VISCOLOR.TXT`, use default file or derive simple visualizer colors from predicted atlas.
- [ ] For cursor files, copy default assets; do not train them.

Magenta/transparency verification before training:

- [ ] Implement `scripts/07_verify_magenta.py`.
- [ ] Load several skins containing exact RGB `#FF00FF` pixels in `CBUTTONS`, `TITLEBAR`, `PLEDIT`, and other slots.
- [ ] Render them in Cranamp and inspect whether `#FF00FF` is rendered as opaque purple or treated as transparent/keyed.
- [ ] Write `configs/magenta_policy.json` using the per-slot schema below.
- [ ] If Cranamp treats magenta as technical/transparent in a slot, set that slot to `true`.
- [ ] If Cranamp renders magenta as opaque purple in a slot, set that slot to `false`.
- [ ] Do not assume magenta is technical globally; verify per slot because Cranamp may treat different files/slots differently.

Expected format of `configs/magenta_policy.json`:

```json
{
  "default": false,
  "per_slot": {
    "MAIN": false,
    "TITLEBAR": false,
    "CBUTTONS": true,
    "SHUFREP": true,
    "MONOSTER": false,
    "PLAYPAUS": false,
    "EQMAIN": false,
    "PLEDIT": true,
    "POSBAR": false,
    "VOLUME": false,
    "BALANCE": false
  }
}
```

- [ ] Treat the schema above as example output; final booleans must come from `scripts/07_verify_magenta.py` and Cranamp behavior.

Expected format of `configs/export_profile_classic.json`:

```json
{
  "MAIN.bmp":     {"slot": "MAIN",     "w": 275, "h": 116},
  "EQMAIN.bmp":   {"slot": "EQMAIN",   "w": 275, "h": 315},
  "PLEDIT.bmp":   {"slot": "PLEDIT",   "w": 280, "h": 186},
  "CBUTTONS.bmp": {"slot": "CBUTTONS", "w": 136, "h": 36}
}
```

- [ ] Treat the numbers above as example placeholders until `cranamp-cli dump-classic-spec` fills final values.

Export rule:

```python
bmp = atlas[
  slot.y : slot.y + export_profile[file].h,
  slot.x : slot.x + export_profile[file].w
]
save_bmp(file, bmp)
```


## 7.5 State region configuration

- [ ] Create `configs/state_regions_v1.json`.
- [ ] Use this config only for metrics/debug/escalation; do not make it a required training target in V0.
- [ ] Define state regions in atlas-slot-local pixel coordinates, not normalized coordinates.
- [ ] Derive exact rectangles from Cranamp’s skin loader/render constants; do not guess state-strip offsets.
- [ ] Schema root keys are slot names: `CBUTTONS`, `SHUFREP`, `PLAYPAUS`, `VOLUME`, `BALANCE`, `POSBAR`.
- [ ] Each slot entry contains named state regions with arrays of rectangles.
- [ ] Rectangle format is `[x0, y0, x1, y1]` in slot-local pixels, half-open range.
- [ ] Include at least these region groups when Cranamp exposes them: `normal`, `pressed`, `active`, `inactive`, `hidden`, `visible_current_state`.
- [ ] For `CBUTTONS`, define pressed/normal transport-button state regions from the actual `CBUTTONS.bmp` strip layout.
- [ ] For `SHUFREP`, define toggle on/off regions for shuffle/repeat/EQ/playlist buttons if Cranamp renders them from distinct regions.
- [ ] For `PLAYPAUS`, define play/pause indicator state regions if Cranamp renders them from distinct regions.
- [ ] For `VOLUME`, `BALANCE`, and `POSBAR`, define movable-thumb/frame regions used for state coverage metrics.
- [ ] If a state-bearing slot has no distinct atlas region in Cranamp, set that state region list empty rather than guessing.
- [ ] `scripts/04_check_dataset.py` must validate that every state-region rectangle lies inside the slot capacity rectangle.
- [ ] `eval_pipeline.py` must use `state_regions_v1.json` for pressed/normal/hidden-state metrics.

Example schema shape, with placeholder rectangles only:

```json
{
  "CBUTTONS": {
    "normal": [[0, 0, 136, 18]],
    "pressed": [[0, 18, 136, 36]]
  },
  "SHUFREP": {
    "normal": [],
    "pressed": []
  }
}
```

- [ ] Replace the placeholder example with Cranamp-derived rectangles before using state-region metrics.

## 8. Atlas packing script

- [ ] Implement `scripts/00_scan_skins.py`.
- [ ] Implement `scripts/01_pack_skins.py`.
- [ ] Scan `skins_raw/` for `.wsz` files and skin directories.
- [ ] Extract `.wsz` files into a temporary normalized directory.
- [ ] Normalize all BMP filenames case-insensitively.
- [ ] Load BMPs through Pillow.
- [ ] Convert BMPs to RGB.
- [ ] Pack available BMPs into `1024x1024` atlas slots.
- [ ] Paste each BMP into its slot at top-left.
- [ ] Write atlas PNG as RGB.
- [ ] Write atlas mask PNG as grayscale `L`, with `255` for real pasted source BMP pixels and `0` elsewhere.
- [ ] Write per-skin slot weight vector `.f32` for loss weighting and missing optional slot handling.
- [ ] Write per-skin metadata JSON containing original BMP sizes and source paths.
- [ ] If BMP exists but exceeds slot capacity, mark only that slot invalid rather than rejecting the whole skin.
- [ ] If optional BMP is missing, paste a Cranamp/default asset and set that slot’s per-skin weight multiplier to `0.25`.
- [ ] If `MAIN.bmp` is missing and Cranamp cannot render the skin, reject the skin.
- [ ] If `BALANCE.bmp` is missing, do not reject the skin; use default asset and reduced slot weight.
- [ ] If `POSBAR.bmp` is missing, do not reject the skin; use default asset and reduced slot weight.
- [ ] If `SHUFREP.bmp` is missing, do not reject the skin; use default asset and reduced slot weight.
- [ ] Set `TEXT` and `NUMBERS` loss to zero regardless of presence.
- [ ] Write `data_v0/valid_skins.csv`.

Command:

```bash
python scripts/01_pack_skins.py \
  --skins-raw skins_raw \
  --atlas-profile configs/atlas_v1.json \
  --export-profile configs/export_profile_classic.json \
  --default-skin assets/default_skin \
  --out data_v0
```

Outputs:

```text
data_v0/atlases/{skin_id}.png
data_v0/atlases/{skin_id}.mask.png
data_v0/atlases/{skin_id}.slot_weight.f32
data_v0/atlases/{skin_id}.meta.json
data_v0/valid_skins.csv
```

## 9. Component rect vector

- [ ] Every rendered sample must have a rect file.
- [ ] Rect file extension is `.f32`.
- [ ] Rect file dtype is little-endian float32.
- [ ] Rect file shape is `[80, 5]`.
- [ ] Rect file is flattened as `80 * 5` float32 values.
- [ ] Each rect entry is `[x0_norm, y0_norm, x1_norm, y1_norm, visible]`.
- [ ] Normalize x coordinates by `INPUT_W = 960`.
- [ ] Normalize y coordinates by `INPUT_H = 1728`.
- [ ] Set `visible = 1.0` if component was drawn and visible.
- [ ] Set `visible = 0.0` if component was absent, hidden, clipped away, or not drawn.
- [ ] Clip rects to canvas bounds before normalization.
- [ ] If clipped visible area is less than `4` pixels, set `visible = 0.0` and rect to zero.
- [ ] Reserve rect classes `60-79` for future extension.
- [ ] Use 80 rect classes from day one to avoid architectural changes later.

Component IDs:

```text
0   main_window
1   main_titlebar
2   main_display
3   main_song_title
4   main_vis_area
5   main_posbar
6   main_transport_row
7   main_prev_button
8   main_play_button
9   main_pause_button
10  main_stop_button
11  main_next_button
12  main_eject_button
13  main_volume_block
14  main_volume_thumb
15  main_balance_block
16  main_balance_thumb
17  main_shuffle_button
18  main_repeat_button
19  main_eq_toggle
20  main_pl_toggle
21  main_mono_stereo
22  main_playpause_indicator
23  main_window_buttons
24  eq_window
25  eq_titlebar
26  eq_graph
27  eq_on_auto_block
28  eq_preamp_slider
29  eq_sliders_group
30  eq_band_60
31  eq_band_170
32  eq_band_310
33  eq_band_600
34  eq_band_1k
35  eq_band_3k
36  eq_band_6k
37  eq_band_12k
38  eq_band_14k
39  eq_band_16k
40  eq_presets_button
41  eq_close_button
42  playlist_window
43  playlist_titlebar
44  playlist_text_area
45  playlist_selected_row
46  playlist_scrollbar_track
47  playlist_scrollbar_thumb
48  playlist_bottom_bar
49  playlist_add_button
50  playlist_rem_button
51  playlist_sel_button
52  playlist_misc_button
53  playlist_list_button
54  playlist_minicontrols
55  playlist_time_box
56  playlist_resize_grip
57  playlist_close_button
58  playlist_shade_button
59  playlist_window_buttons
60-79 reserved
```

- [ ] Cranamp should output exact per-band EQ rects for classes `30-39`.
- [ ] GeoNet should output all 80 classes, but V0 should not require detecting individual EQ band rects as independent heatmap positives.
- [ ] Use `eq_sliders_group` as the primary detected geometry for EQ sliders in V0.
- [ ] Derive `eq_band_60` through `eq_band_16k` rects by equal subdivision inside `eq_sliders_group` during postprocessing.
- [ ] Keep the explicit EQ band labels in saved data for validation and future experiments.
- [ ] Provide config flag `--train-eq-band-heatmaps`; default is `false` in V0.
- [ ] When `--train-eq-band-heatmaps=false`, exclude classes 30-39 from heatmap positive and negative loss; do not let them become background-only channels that bias training.

## 10. State vector

- [ ] Every rendered sample must have a state file.
- [ ] State file extension is `.f32`.
- [ ] State file dtype is little-endian float32.
- [ ] State file shape is `[32]`.
- [ ] State entries are normalized floats.
- [ ] Reserve state entries `27-31` for future extension.

State IDs:

```text
0   pressed_transport_button_norm
1   volume_pos_0_1
2   balance_pos_0_1
3   posbar_pos_0_1
4   shuffle_on_0_1
5   repeat_on_0_1
6   eq_on_0_1
7   eq_auto_0_1
8   eq_preamp_0_1
9   eq_band_60_0_1
10  eq_band_170_0_1
11  eq_band_310_0_1
12  eq_band_600_0_1
13  eq_band_1k_0_1
14  eq_band_3k_0_1
15  eq_band_6k_0_1
16  eq_band_12k_0_1
17  eq_band_14k_0_1
18  eq_band_16k_0_1
19  playlist_scroll_0_1
20  playlist_selected_row_0_1
21  main_scale_x_norm
22  main_scale_y_norm
23  eq_scale_x_norm
24  eq_scale_y_norm
25  playlist_scale_x_norm
26  playlist_scale_y_norm
27-31 reserved
```

Pressed transport button normalization:

```text
no button pressed: 0.0
prev:              1/7
play:              2/7
pause:             3/7
stop:              4/7
next:              5/7
eject:             6/7
reserved:          1.0
```

## 11. Visible atlas mask

- [x] Every rendered sample must include `visible_atlas_mask.png`.
- [x] Visible atlas mask shape is `1024x1024`.
- [x] Visible atlas mask mode is grayscale `L`.
- [x] Mask value is `255` if the atlas pixel was sampled/drawn into the rendered view.
- [x] Mask value is `0` otherwise.
- [x] When Cranamp blits a sprite/asset pixel to the rendered view, it must mark the corresponding atlas pixel visible.
- [ ] This requires modifying Cranamp renderer internals; budget this as real Cranamp work, not a dataset-script detail.
- [ ] Implement provenance by carrying `(slot_id, atlas_x, atlas_y)` through each draw call or by a dedicated tracking pass that draws atlas IDs into a side buffer.
- [ ] Prefer a separate `u32` side buffer over per-pixel writes inside the RGB buffer so provenance tracking can be optimized independently.
- [x] Use the same provenance encoding as Section 4: `0` means no atlas source, otherwise `1 + (slot_id << 20) + (atlas_y << 10) + atlas_x`.
- [x] Verify visible mask with unit/integration tests before using it in SlotNet loss.
- [ ] Use visible atlas mask because a single render does not show every button state, slider frame, titlebar state, or hidden source pixel.
- [x] During SlotNet loss, compute `effective_loss_mask = atlas_mask * (0.25 + 0.75 * visible_atlas_mask)` after normalizing visible mask to `0..1`.
- [x] Hidden atlas pixels still train through base `0.25` weight.
- [x] Visible atlas pixels receive full `1.0` weight.

## 12. Randomized Cranamp dataset generation

- [x] Generate views offline and save them.
- [ ] Before generating V0/V1, run `scripts/06_benchmark_cranamp.py` and record render throughput.
- [ ] Use multiple Cranamp worker processes if render throughput is slower than `0.50 s/render` or CPU has unused cores.
- [ ] Do not generate views on the fly during model training.
- [x] Implement `scripts/02_render_dataset.py` to call `cranamp-cli render-random`.
- [x] Use deterministic seed per `(skin_id, variant_id)`.
- [ ] Use `seed = stable_hash(skin_id) * 1000003 + variant_id`.
- [x] Save all generated files to disk.
- [ ] Keep loose files for SMOKE/V0 unless dataloader profiling shows GPU starvation.
- [ ] If GPU utilization is below `80%` due to file I/O, run optional `scripts/08_pack_dataset_shards.py` to pack samples into sequential shards.
- [x] Save `params.json` for replay and validation.
- [x] Save exact rects from Cranamp.
- [x] Save exact state vector from Cranamp.
- [x] Save visible atlas mask from Cranamp.
- [ ] Do not mutate skin style in V0.
- [ ] Do not create AI/diffusion-generated views in V0.
- [ ] Do not rotate windows in V0.
- [ ] Include uniform scaling.
- [ ] Include horizontal scaling.
- [ ] Include vertical scaling.
- [ ] Include upscaling.
- [ ] Include downscaling.
- [ ] Include x/y displacement.
- [ ] Include component displacement.
- [ ] Include component local scaling.
- [ ] Include random seeker/slider positions.
- [ ] Include random button pressed states.
- [ ] Include random EQ states and bands.
- [ ] Include random playlist scroll and selected row.

Dataset generation command:

```bash
python scripts/02_render_dataset.py \
  --valid-skins data_v0/valid_skins.csv \
  --variants 16 \
  --canvas-w 960 \
  --canvas-h 1728 \
  --state-balanced false \
  --cranamp-cli ./cranamp_cli/cranamp-cli \
  --out data_v0
```

Per-sample outputs:

```text
data_v0/views/{skin_id}_{variant_id}.png
data_v0/rects/{skin_id}_{variant_id}.f32
data_v0/states/{skin_id}_{variant_id}.f32
data_v0/visible_masks/{skin_id}_{variant_id}.png
data_v0/params/{skin_id}_{variant_id}.json
```

### 12.1 Render canvas

- [x] Use canvas width `960` (raised from 941 to be divisible by 32).
- [x] Use canvas height `1728` (raised from 1672 to be divisible by 32).
- [x] Use RGB output.
- [x] Use solid dark background RGB `(16,16,16)` or Cranamp default background.
- [x] Keep background choice deterministic and logged in params.

### 12.2 Window scale and placement distributions

- [x] Sample global scale up to the top-left 960x1728 fit, with some samples leaving blank padding.
- [x] Use one uniform window scale for the whole main/EQ/playlist stack.
- [x] Keep `main_x = 0` and `main_y = 0`.
- [x] Keep `eq_x = main_x` and `eq_y = main_y + 116 * scale`.
- [x] Keep `pl_x = main_x`.
- [x] Keep `pl_y = eq_y + 116 * scale`.
- [ ] Clip windows to canvas if needed and mark component visibility accordingly.

### 12.3 Per-component local displacement and scale distributions

- [x] Sample movement/scale modes for individual transport buttons: move, scalex, scaley, scalexy, and move+scale combinations.
- [x] Sample movement/scale modes for shuffle/repeat/EQ/PL toggles independently.
- [x] Sample movement/scale modes for volume, balance, posbar, EQ sliders, and playlist scrollbar.
- [x] Fill each original transformed-control rect with the average of four outside edge pixels before drawing the transformed control.
- [ ] Sample `eq_graph_dx ~ Uniform(-12, 12)`.
- [ ] Sample `eq_graph_dy ~ Uniform(-12, 12)`.
- [ ] Sample `eq_graph_sx ~ Uniform(0.88, 1.16)`.
- [ ] Sample `eq_graph_sy ~ Uniform(0.88, 1.16)`.
- [ ] Sample `playlist_text_dx ~ Uniform(-16, 16)`.
- [ ] Sample `playlist_text_dy ~ Uniform(-16, 16)`.
- [ ] Sample `playlist_text_sx ~ Uniform(0.88, 1.18)`.
- [ ] Sample `playlist_text_sy ~ Uniform(0.88, 1.18)`.
- [ ] Sample `playlist_bottom_dx ~ Uniform(-16, 16)`.
- [ ] Sample `playlist_bottom_dy ~ Uniform(-16, 16)`.
- [ ] Sample `playlist_bottom_sx ~ Uniform(0.85, 1.20)`.
- [ ] Sample `playlist_bottom_sy ~ Uniform(0.85, 1.20)`.

### 12.4 Dynamic state randomization

- [ ] Sample `pressed_transport_button = Choice([-1, 0, 1, 2, 3, 4, 5])`.
- [ ] Sample `volume_pos = Uniform(0.0, 1.0)`.
- [ ] Sample `balance_pos = Uniform(-1.0, 1.0)`.
- [ ] Sample `posbar_pos = Uniform(0.0, 1.0)`.
- [ ] Sample `shuffle_on = Bernoulli(0.5)`.
- [ ] Sample `repeat_on = Bernoulli(0.5)`.
- [ ] Sample `eq_on = Bernoulli(0.5)`.
- [ ] Sample `eq_auto = Bernoulli(0.5)`.
- [ ] Sample `eq_preamp = Uniform(-12.0, 12.0)`.
- [ ] Sample each of `eq_bands[10] = Uniform(-12.0, 12.0)`.
- [ ] Sample `playlist_scroll = Uniform(0.0, 1.0)`.
- [ ] Sample `playlist_selected_row = RandomInt(0, 16)`.
- [x] Render deterministic stub playlist song names.
- [x] Render deterministic main-window histogram bars.
- [x] Render visible seek progress in addition to the seek thumb.
- [ ] Log state coverage counts per skin for pressed transport states, shuffle/repeat states, volume/balance/posbar positions, and EQ bands.
- [x] Implement optional `--state-balanced` mode but keep it disabled by default in V0.
- [x] Expose `--state-balanced` in both `cranamp-cli render-random` and `scripts/02_render_dataset.py`.
- [ ] In `--state-balanced` mode, guarantee each transport pressed state including `-1` appears at least three times per skin when `--variants >= 32`.
- [ ] In `--state-balanced` mode, enumerate shuffle/repeat/EQ/playlist toggle states across variants rather than sampling all toggles independently.
- [ ] In `--state-balanced` mode, distribute volume/balance/posbar positions across low/mid/high buckets rather than pure uniform sampling.
- [ ] Do not enable `--state-balanced` for V0 by default; enable only if rare-state metrics fail the numeric threshold defined in Section 30.

## 13. Dataset splits

- [x] Implement `scripts/03_make_splits.py`.
- [x] Split by `skin_id`, never by variant.
- [x] Ensure no skin appears in more than one split.
- [x] Use default train split `0.80`.
- [x] Use default validation split `0.10`.
- [x] Use default test split `0.10`.
- [x] Write `data_v0/train.csv`.
- [x] Write `data_v0/val.csv`.
- [x] Write `data_v0/test.csv`.

Command:

```bash
python scripts/03_make_splits.py \
  --data data_v0 \
  --train 0.80 \
  --val 0.10 \
  --test 0.10 \
  --split-by skin_id
```

CSV columns:

```text
skin_id,
variant_id,
view_png,
rects_f32,
state_f32,
visible_mask_png,
atlas_png,
atlas_mask_png,
slot_weight_f32,
meta_json,
params_json
```

## 14. Dataset scale plan

- [ ] Smoke dataset: `20 skins x 4 variants = 80 samples`.
- [ ] V0 dataset: `500 skins x 16 variants = 8,000 samples`.
- [ ] V1 dataset: `5,000 skins x 16 variants = 80,000 samples`.
- [ ] Full dataset: `10,000 skins x 16 variants = 160,000 samples`.
- [ ] Do not run Full before Smoke and V0 pass.
- [ ] Do not create diffusion/img2img data before V0 pass.

## 15. Dataset checking and debug contact sheets

- [x] Implement `scripts/04_check_dataset.py`.
- [x] Verify every CSV path exists.
- [x] Verify every view is `960x1728` RGB.
- [x] Verify every rect file has exactly `80*5` float32 values.
- [x] Verify every state file has exactly `32` float32 values.
- [x] Verify every atlas is `1024x1024` RGB.
- [x] Verify every atlas mask is `1024x1024` grayscale.
- [x] Verify every visible mask is `1024x1024` grayscale.
- [x] Verify rect coordinates are in `[0,1]` or zeroed when invisible.
- [x] Verify visible flags are `0.0` or `1.0`.
- [x] Verify slot weights are present and valid.
- [x] Generate debug contact sheet of 32 random samples with rect overlays.
- [ ] Generate debug atlas/mask contact sheet of 16 random atlases.
- [x] Fail fast on shape mismatch.

Command:

```bash
python scripts/04_check_dataset.py --data data_v0
```

## 16. Model 1: GeoNet80 task

- [x] Implement `models/geonet80.py`.
- [x] GeoNet80 input is `view image [B, 3, 1728, 960]`.
- [x] GeoNet80 output includes CenterNet-style heatmaps.
- [x] GeoNet80 output includes width/height maps.
- [x] GeoNet80 output includes offset maps.
- [x] GeoNet80 output includes state vector `[B, 32]`.
- [x] GeoNet80 decoded output must be `rects [B, 80, 5]` and `states [B, 32]`.
- [x] Use CenterNet-style fixed-class detector instead of YOLO.
- [x] Use 80 fixed component classes.
- [ ] Use stride-4 detection head to reduce small-component/EQ-band localization noise.
- [x] Use EQ-band postprocessing from `eq_sliders_group` as the V0 default.
- [ ] Do not spend GeoNet capacity on tightly packed individual EQ band heatmaps unless later experiments justify it.

## 17. GeoNet80 exact architecture

- [ ] Input tensor: `B x 3 x 1728 x 960`.
- [ ] Backbone: ResNet34 without classification head.
- [ ] Use feature `C2: B x 64 x 432 x 240`, stride 4.
- [ ] Use feature `C3: B x 128 x 216 x 120`, stride 8.
- [ ] Use feature `C4: B x 256 x 108 x 60`, stride 16.
- [ ] Use feature `C5: B x 512 x 54 x 30`, stride 32.
- [ ] Implement FPN lateral `lat5 = Conv1x1(512 -> 128)(C5)`.
- [ ] Implement FPN lateral `lat4 = Conv1x1(256 -> 128)(C4)`.
- [ ] Implement FPN lateral `lat3 = Conv1x1(128 -> 128)(C3)`.
- [ ] Implement FPN lateral `lat2 = Conv1x1(64 -> 128)(C2)`.
- [ ] Build `P = lat2 + upsample(lat3, 432x240) + upsample(lat4, 432x240) + upsample(lat5, 432x240)`.
- [ ] Use nearest upsampling in FPN.
- [ ] Apply `Conv3x3(128 -> 128), ReLU` to P.
- [ ] Apply second `Conv3x3(128 -> 128), ReLU` to P.
- [ ] Detection heads operate at stride 4, output spatial size `432x240`.
- [ ] Heatmap head: `Conv3x3(128 -> 128), ReLU; Conv1x1(128 -> 80); Sigmoid`.
- [ ] Heatmap output shape: `B x 80 x 320 x 192`.
- [ ] Width/height head: `Conv3x3(128 -> 128), ReLU; Conv1x1(128 -> 160); Softplus`.
- [ ] Width/height output shape: `B x 160 x 320 x 192`.
- [ ] Width/height semantics: 2 channels per component, normalized width and normalized height.
- [ ] Offset head: `Conv3x3(128 -> 128), ReLU; Conv1x1(128 -> 160); Tanh`.
- [ ] Offset output shape: `B x 160 x 320 x 192`.
- [ ] Offset semantics: 2 channels per component, subcell dx and dy.
- [ ] Do not use `GlobalAveragePool(C5)` for state regression in V0.
- [ ] Implement local component-anchored state regression from FPN feature map `P`.
- [ ] State head samples a local `128`-channel feature from `P` at the center of the relevant component rect for each state.
- [ ] Do not use single-pixel `P[:, :, cy, cx]` lookup for state features.
- [ ] Use 3x3 average pooling around the state anchor center on `P` to obtain the 128-channel sampled feature.
- [ ] During GeoNet training, state head uses ground-truth rect centers for local feature sampling.
- [ ] During GeoNet training, add state-anchor jitter before 3x3 pooling.
- [ ] Compute anchor width in grid cells: `w_cells = max(1e-6, rect_w_norm * 192)`.
- [ ] Compute anchor height in grid cells: `h_cells = max(1e-6, rect_h_norm * 320)`.
- [ ] Compute jitter limits: `jx_max = min(2.0, 0.25 * w_cells)`, `jy_max = min(2.0, 0.25 * h_cells)`.
- [ ] Sample `jx ~ Uniform(-jx_max, jx_max)` and `jy ~ Uniform(-jy_max, jy_max)` in stride-4 grid cells for each state anchor.
- [ ] This keeps jitter at roughly ±8 input pixels for large anchors while preventing tiny anchors from being jittered by their entire width.
- [ ] Clamp jittered state anchor centers to valid `P` coordinates before 3x3 pooling.
- [ ] During GeoNet inference, state head uses decoded/predicted rect centers for local feature sampling and uses no random jitter.
- [ ] GeoNet inference is explicitly two-pass: first run backbone/FPN and detection heads, decode rects, then sample state features from `P` at predicted anchor centers and decode states.
- [ ] State head uses `nn.Embedding(32, 16)` for state index embedding.
- [ ] State head input per state is `[sampled_feature_128, state_embedding_16, anchor_rect_5]`, total `149` channels.
- [ ] Shared state MLP: `Linear(149 -> 64), ReLU; Linear(64 -> 1); Sigmoid`.
- [ ] State output shape: `B x 32`.

## 18. GeoNet80 labels

- [ ] For component `k`, compute `cx = (x0 + x1) / 2`.
- [ ] For component `k`, compute `cy = (y0 + y1) / 2`.
- [ ] For component `k`, compute `w = x1 - x0`.
- [ ] For component `k`, compute `h = y1 - y0`.
- [ ] Compute `grid_x = cx * 192`.
- [ ] Compute `grid_y = cy * 320`.
- [x] Heatmap has 80 channels.
- [ ] Draw Gaussian center with radius `2` cells for visible components.
- [ ] Store `wh[k] = [w, h]` at center location.
- [ ] Store `offset[k] = [grid_x - floor(grid_x), grid_y - floor(grid_y)]` at center location.
- [ ] Ignore invisible components for heatmap positives, wh loss, and offset loss.
- [ ] Exclude reserved classes 60-79 from both positive and negative heatmap loss.
- [ ] Exclude EQ band classes 30-39 from both positive and negative heatmap loss when `--train-eq-band-heatmaps=false`.
- [ ] Use state vector as direct regression target.
- [ ] Use this state-to-anchor map for local feature sampling: state 0 -> `main_transport_row`; 1 -> `main_volume_block`; 2 -> `main_balance_block`; 3 -> `main_posbar`; 4 -> `main_shuffle_button`; 5 -> `main_repeat_button`; 6 -> `eq_on_auto_block`; 7 -> `eq_on_auto_block`; 8 -> `eq_preamp_slider`; 9-18 -> derived/predicted EQ band rects 30-39; 19 -> `playlist_scrollbar_track`; 20 -> `playlist_text_area`; 21-22 -> `main_window`; 23-24 -> `eq_window`; 25-26 -> `playlist_window`; 27-31 -> `main_window`.
- [x] Implement one shared function `derive_eq_band_rects(eq_sliders_group_rect)` used in both training and inference.
- [x] `derive_eq_band_rects` must split the group rect into 10 equal-width horizontal subdivisions.
- [x] `derive_eq_band_rects` must use the full vertical extent of `eq_sliders_group_rect`.
- [x] `derive_eq_band_rects` must apply no expansion, no padding, and no hidden magic constants.
- [ ] For band `i` in `0..9`, compute `x0_i = x0 + (x1 - x0) * i / 10` and `x1_i = x0 + (x1 - x0) * (i + 1) / 10`; use original `y0,y1`.
- [ ] If a state anchor is invisible or missing, feed zero feature plus anchor rect `[0,0,0,0,0]` for that state.

## 19. GeoNet80 loss

- [x] Implement CenterNet focal heatmap loss.
- [ ] Positive heatmap term: `-((1 - p) ** 2) * log(p)`.
- [ ] Negative heatmap term: `-((1 - y) ** 4) * (p ** 2) * log(1 - p)`.
- [ ] Normalize heatmap loss by number of visible components.
- [x] Implement smooth L1 loss for width/height at target centers.
- [x] Implement smooth L1 loss for offsets at target centers.
- [x] Implement smooth L1 loss for state vector.
- [ ] Use `L_geonet = 1.0 * center_focal_loss + 5.0 * smooth_l1_wh + 2.0 * smooth_l1_offset + 1.0 * smooth_l1_state`.

## 20. GeoNet80 training augmentation

- [ ] Apply geometric letterbox/pad augmentation before photometric augmentations with probability `p = 0.30`.
- [ ] Letterbox augmentation creates a new `960x1728` canvas with dark/black padding, rescales the whole view into a random sub-rectangle, and updates rect labels accordingly.
- [ ] Letterbox content scale x sampled from `[0.80, 1.00]`.
- [ ] Letterbox content scale y sampled from `[0.80, 1.00]`.
- [ ] Letterbox paste x sampled uniformly from available horizontal padding.
- [ ] Letterbox paste y sampled uniformly from available vertical padding.
- [ ] Clip all updated rects to the canvas after letterbox augmentation.
- [ ] Apply photometric augmentations to training input views after any letterbox augmentation.
- [ ] Do not change rect labels with photometric augmentations.
- [ ] Brightness factor sampled from `[0.88, 1.12]`.
- [ ] Contrast factor sampled from `[0.88, 1.12]`.
- [ ] Saturation factor sampled from `[0.88, 1.12]`.
- [ ] Hue offset sampled from `[-0.03, 0.03]`.
- [ ] Gaussian noise probability `p = 0.25`.
- [ ] Gaussian noise sigma sampled from `[0.0, 0.015]`.
- [ ] Gaussian blur probability `p = 0.15`.
- [ ] Gaussian blur kernel size `3`.
- [ ] Gaussian blur sigma sampled from `[0.0, 0.8]`.
- [ ] JPEG roundtrip probability `p = 0.25`.
- [ ] JPEG quality sampled as integer from `[70, 95]`.
- [ ] Use these augmentations to improve generalization beyond clean Cranamp output.

## 21. GeoNet80 training script

- [x] Implement `train_geonet.py`.
- [x] Use AdamW optimizer.
- [x] Use learning rate `2e-4`.
- [x] Use weight decay `1e-4`.
- [ ] Use cosine decay scheduler with 5% warmup.
- [ ] Use AMP/mixed precision.
- [x] Use gradient clipping with `max_norm = 1.0`.
- [ ] Default batch size `4`.
- [ ] Allow batch size `8` if RTX 4090 memory allows.
- [ ] If GeoNet OOMs at batch `4`, use micro-batch `2` with `--grad-accum-steps 2` to keep effective batch `4`.
- [ ] Do not silently reduce effective batch below `4` without logging it in `config.yaml`.
- [ ] Train default `80` epochs.
- [ ] Save checkpoint every epoch.
- [x] Save `last.safetensors`.
- [x] Save `best.safetensors`.
- [x] Save `config.yaml`.
- [x] Save `metrics.jsonl`.
- [ ] Save visual rect overlay debug grids every validation epoch.

Training command:

```bash
python train_geonet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --image-h 1728 \
  --image-w 960 \
  --components 80 \
  --batch 4 \
  --grad-accum-steps 1 \
  --epochs 80 \
  --lr 2e-4 \
  --amp \
  --out runs/geonet80_v0
```

## 22. GeoNet80 validation/checkpoint metrics

- [ ] Decode predicted rects from heatmap peaks.
- [ ] Compute mean IoU for visible rects.
- [ ] Compute median center error in pixels.
- [ ] Compute 95th percentile center error in pixels.
- [ ] Compute mean IoU for main/eq/playlist windows.
- [ ] Compute mean IoU for small controls.
- [ ] Compute state L1.
- [ ] Save best checkpoint by `mean_rect_iou_visible`.
- [ ] Report `val_score = mean_rect_iou_visible - 0.1 * mean_center_error_pixels - val_state_l1` as secondary score.
- [ ] Include rect overlay images in debug output.

## 23. Model 2: SlotNetV1 task

**Correction (2026-05-16):** The original §23-31 design pre-cropped the input view to each slot's predicted source rect before feeding SlotNet. This was an incorrect simplification that cost real quality:

- It discarded global style/palette context — the MAIN decoder couldn't see EQMAIN colors, so the model couldn't make sister slots match.
- It made the pipeline brittle to small GeoNet rect errors — a few-pixel drift made the crop swallow letterbox padding, and the smoke training visibly failed in exactly this mode.
- The compute saving was modest (~half-canvas total area summed across slots) and disappears once the encoder is shared across slots.

V2 fix: SlotNet runs a **shared encoder over the full view**, then **per-slot decoders** use **ROI-aligned encoder features** at the slot's atlas dimensions plus a **globally pooled style vector**. The decoder still outputs `[B, 7, slot_h, slot_w]` so the export path is unchanged. The roadmap text below is kept verbatim for historical context; treat `models/slotnet_v2.py` as the correct implementation.



- [x] Implement `models/slotnet_v1.py`.
- [x] SlotNet input includes full view image.
- [x] SlotNet input includes rects `[B, 80, 5]`.
- [x] SlotNet input includes state vector `[B, 32]`.
- [x] SlotNet input includes `slot_id`.
- [x] SlotNet crops source view region for the selected slot using rects.
- [x] SlotNet output is one slot bitmap, not the full atlas at once.
- [x] SlotNet output shape is `[B, 7, slot_h, slot_w]`.
- [x] Output channels `0..2` are RGB via sigmoid.
- [x] Output channels `3..6` are special logits.
- [ ] Special class `0` means normal RGB.
- [ ] Special class `1` means forced magenta.
- [ ] Special class `2` means forced black.
- [ ] Special class `3` means ignored padding.
- [ ] Train RGB in V0 and train magenta special class only if `configs/magenta_policy.json` enables it.
- [ ] Keep other special classes as placeholders.
- [ ] Use a shared U-Net for all slots with slot embeddings, not separate models per slot.
- [ ] Use grouped-by-slot batching because slot sizes differ.

## 24. Slot source mapping

- [x] Create `configs/slot_sources_v1.json`.
- [x] Map `MAIN` source to `main_window`, expand `0.00`.
- [x] Map `TITLEBAR` source to `union(main_titlebar, main_window_buttons)`, expand `0.08`.
- [x] Map `CBUTTONS` source to `main_transport_row`, expand `0.12`.
- [x] Map `SHUFREP` source to `union(main_shuffle_button, main_repeat_button, main_eq_toggle, main_pl_toggle)`, expand `0.12`.
- [x] Map `MONOSTER` source to `main_mono_stereo`, expand `0.12`.
- [x] Map `PLAYPAUS` source to `main_playpause_indicator`, expand `0.12`.
- [x] Map `EQMAIN` source to `eq_window`, expand `0.00`.
- [x] Map `PLEDIT` source to `playlist_window`, expand `0.00`.
- [x] Map `POSBAR` source to `main_posbar`, expand `0.12`.
- [x] Map `VOLUME` source to `main_volume_block`, expand `0.12`.
- [x] Map `BALANCE` source to `main_balance_block`, expand `0.12`.
- [ ] If a source rect is missing, fall back to its parent window rect.
- [ ] If parent window rect is also missing, use zero crop and disable slot loss for that sample.
- [ ] `union(...)` means bounding rectangle around listed visible component rects, expanded by the configured fraction.

## 25. SlotNet rect jitter robustness

- [ ] Train SlotNet with rect jitter so it survives GeoNet errors at inference.
- [ ] During SlotNet training, jitter the selected source rect before cropping.
- [ ] Compute rect center `(cx, cy)` and size `(w, h)`.
- [ ] Sample `cx_jitter = Uniform(-0.08 * w, 0.08 * w)`.
- [ ] Sample `cy_jitter = Uniform(-0.08 * h, 0.08 * h)`.
- [ ] Sample `sx_jitter = exp(Uniform(-0.12, 0.12))`.
- [ ] Sample `sy_jitter = exp(Uniform(-0.12, 0.12))`.
- [ ] Set `new_cx = cx + cx_jitter`.
- [ ] Set `new_cy = cy + cy_jitter`.
- [ ] Set `new_w = w * sx_jitter`.
- [ ] Set `new_h = h * sy_jitter`.
- [ ] Clip jittered rect to input canvas.
- [ ] Low-priority crop drop probability is `p = 0.05`.
- [ ] Low-priority crop drop applies to `MONOSTER`, `PLAYPAUS`, `SHUFREP`, and `POSBAR`.
- [ ] Crop drop action replaces source rect with parent `main_window` rect.
- [ ] SlotNet Stage A uses ground-truth rects plus jitter.
- [ ] SlotNet Stage B uses `--rect-mix-ratio 0.5` by default: 50% ground-truth rects plus jitter and 50% frozen GeoNet-predicted rects plus jitter.
- [ ] Make `--rect-mix-ratio` configurable and log the actual sampled ratio.

## 26. Slot crop operation

- [x] Implement `models/crop.py`.
- [x] Use `torch.nn.functional.grid_sample`.
- [x] For each sample and slot, crop source view over source rect.
- [x] Crop output size equals slot capacity size in `(H, W)` order.
- [x] Add tests for H/W ordering to prevent width-height swaps.
- [ ] `MAIN` crop output size is `128x320`.
- [ ] `TITLEBAR` crop output size is `96x320`.
- [ ] `CBUTTONS` crop output size is `64x160`.
- [ ] `SHUFREP` crop output size is `128x128`.
- [ ] `MONOSTER` crop output size is `32x64`.
- [ ] `PLAYPAUS` crop output size is `32x64`.
- [ ] `EQMAIN` crop output size is `384x320`.
- [ ] `PLEDIT` crop output size is `256x320`.
- [ ] `POSBAR` crop output size is `32x320`.
- [ ] `VOLUME` crop output size is `448x96`.
- [ ] `BALANCE` crop output size is `448x80`.
- [x] Crop tensor shape is `B x 3 x slot_h x slot_w`.
- [ ] Compute source crop size in input-render pixels before resizing: `render_w = (x1 - x0) * INPUT_W`, `render_h = (y1 - y0) * INPUT_H`.
- [ ] Compute raw SlotNet scale ratios: `scale_x = render_w / slot_w`, `scale_y = render_h / slot_h`.
- [x] Convert to log-ratio conditioning values: `log_scale_x = log(max(scale_x, 1e-6))`, `log_scale_y = log(max(scale_y, 1e-6))`.
- [x] Use log-ratios as SlotNet input channels, not raw scale ratios.
- [ ] If rect visibility is zero, set `log_scale_x = 0` and `log_scale_y = 0` and mark the slot loss disabled for that sample if no fallback rect exists.
- [x] Use `align_corners=False` unless tests show a reason to change.
- [x] Add unit test for rect-to-grid conversion.
- [x] Add unit test that an identity/full-image rect crop matches expected resized image.

## 27. SlotNet input channels

- [ ] Concatenate RGB crop with extra conditioning channels.
- [ ] RGB crop contributes 3 channels.
- [ ] Coordinate channels contribute 2 channels, x and y in `[-1, 1]`.
- [ ] Slot embedding contributes 16 channels via `nn.Embedding(18, 16)` broadcast over H/W.
- [ ] State embedding contributes 8 channels via MLP `32 -> 64 -> 8`, broadcast over H/W.
- [ ] Rect channels contribute 5 channels: `x0, y0, x1, y1, visible`, broadcast over H/W.
- [ ] Scale-ratio channels contribute 2 channels: `log_scale_x = log(max(render_w / slot_w, 1e-6))` and `log_scale_y = log(max(render_h / slot_h, 1e-6))`, broadcast over H/W.
- [ ] Total input channels: `3 + 2 + 16 + 8 + 5 + 2 = 36`.
- [ ] Do not feed raw scale ratios unless an ablation explicitly enables `--raw-scale-ratio-channels`; default is log-ratio channels.

## 28. SlotNet exact architecture

- [ ] Input shape is `B x 36 x H x W`.
- [ ] Implement two depth modes in one shared SlotNet: `deep` for slots with `min(H,W) >= 64` and `shallow` for slots with `min(H,W) < 64`.
- [ ] Use `shallow` mode for `MONOSTER`, `PLAYPAUS`, `POSBAR`, and any future slot with height or width below `64`.
- [ ] Use `deep` mode for `MAIN`, `TITLEBAR`, `CBUTTONS`, `SHUFREP`, `EQMAIN`, `PLEDIT`, `VOLUME`, and `BALANCE`.
- [ ] `shallow` mode must skip `Down3` and use a `256`-channel bottleneck to avoid collapsing tiny slots to unusably small spatial maps.
- [ ] `Enc0` first layer: `Conv3x3 36 -> 64, padding=1`.
- [ ] `Enc0` first norm: `GroupNorm groups=8`.
- [ ] `Enc0` first activation: `SiLU`.
- [ ] `Enc0` second layer: `Conv3x3 64 -> 64, padding=1`.
- [ ] `Enc0` second norm: `GroupNorm groups=8`.
- [ ] `Enc0` second activation: `SiLU`.
- [ ] `Down1` first layer: `Conv3x3 stride=2 64 -> 128, padding=1`.
- [ ] `Down1` first norm: `GroupNorm groups=8`.
- [ ] `Down1` first activation: `SiLU`.
- [ ] `Down1` second layer: `Conv3x3 128 -> 128, padding=1`.
- [ ] `Down1` second norm: `GroupNorm groups=8`.
- [ ] `Down1` second activation: `SiLU`.
- [ ] `Down2` first layer: `Conv3x3 stride=2 128 -> 256, padding=1`.
- [ ] `Down2` first norm: `GroupNorm groups=16`.
- [ ] `Down2` first activation: `SiLU`.
- [ ] `Down2` second layer: `Conv3x3 256 -> 256, padding=1`.
- [ ] `Down2` second norm: `GroupNorm groups=16`.
- [ ] `Down2` second activation: `SiLU`.
- [ ] `Down3` first layer: `Conv3x3 stride=2 256 -> 512, padding=1`.
- [ ] `Down3` first norm: `GroupNorm groups=32`.
- [ ] `Down3` first activation: `SiLU`.
- [ ] `Down3` second layer: `Conv3x3 512 -> 512, padding=1`.
- [ ] `Down3` second norm: `GroupNorm groups=32`.
- [ ] `Down3` second activation: `SiLU`.
- [ ] `Bottleneck` first layer: `Conv3x3 512 -> 512, padding=1`.
- [ ] `Bottleneck` first norm: `GroupNorm groups=32`.
- [ ] `Bottleneck` first activation: `SiLU`.
- [ ] `Bottleneck` second layer: `Conv3x3 512 -> 512, padding=1`.
- [ ] `Bottleneck` second norm: `GroupNorm groups=32`.
- [ ] `Bottleneck` second activation: `SiLU`.
- [ ] `Up2` starts with bilinear upsample x2.
- [ ] `Up2` concatenates Down2 skip output.
- [ ] `Up2` first layer: `Conv3x3 768 -> 256, padding=1`.
- [ ] `Up2` first norm: `GroupNorm groups=16`.
- [ ] `Up2` first activation: `SiLU`.
- [ ] `Up2` second layer: `Conv3x3 256 -> 256, padding=1`.
- [ ] `Up2` second norm: `GroupNorm groups=16`.
- [ ] `Up2` second activation: `SiLU`.
- [ ] `Up1` starts with bilinear upsample x2.
- [ ] `Up1` concatenates Down1 skip output.
- [ ] `Up1` first layer: `Conv3x3 384 -> 128, padding=1`.
- [ ] `Up1` first norm: `GroupNorm groups=8`.
- [ ] `Up1` first activation: `SiLU`.
- [ ] `Up1` second layer: `Conv3x3 128 -> 128, padding=1`.
- [ ] `Up1` second norm: `GroupNorm groups=8`.
- [ ] `Up1` second activation: `SiLU`.
- [ ] `Up0` starts with bilinear upsample x2.
- [ ] `Up0` concatenates Enc0 skip output.
- [ ] `Up0` first layer: `Conv3x3 192 -> 64, padding=1`.
- [ ] `Up0` first norm: `GroupNorm groups=8`.
- [ ] `Up0` first activation: `SiLU`.
- [ ] `Up0` second layer: `Conv3x3 64 -> 64, padding=1`.
- [ ] `Up0` second norm: `GroupNorm groups=8`.
- [ ] `Up0` second activation: `SiLU`.
- [ ] Shallow mode `Bottleneck256` first layer: `Conv3x3 256 -> 256, padding=1`.
- [ ] Shallow mode `Bottleneck256` norm: `GroupNorm groups=16`.
- [ ] Shallow mode `Bottleneck256` activation: `SiLU`.
- [ ] Shallow mode `Bottleneck256` second layer: `Conv3x3 256 -> 256, padding=1`.
- [ ] Shallow mode skips `Down3` and skips deep `Up2`.
- [ ] Shallow `Up1` upsamples bottleneck output, concatenates `Down1` skip, applies `Conv3x3 384 -> 128` then `Conv3x3 128 -> 128`.
- [ ] Shallow `Up0` matches deep `Up0`.
- [ ] Output layer: `Conv1x1 64 -> 7`.
- [ ] Apply sigmoid to output channels `0:3` to produce RGB.
- [ ] Leave output channels `3:7` as special logits.

## 29. SlotNet target and masks

- [x] Implement `models/atlas.py`.
- [x] Load target atlas PNG.
- [x] Load target atlas mask PNG.
- [x] Load visible atlas mask PNG for the sample.
- [x] Crop target slot from target atlas using `atlas_v1.json` slot rectangle.
- [x] Crop atlas mask from atlas mask using same slot rectangle.
- [x] Crop visible atlas mask from visible mask using same slot rectangle.
- [x] Load per-skin slot weight vector from `.slot_weight.f32`.
- [x] Compute slot weight as `atlas_v1 slot loss_weight * per_skin_slot_weight[slot_id]`.
- [x] Normalize RGB targets to `[0,1]`.
- [x] Normalize masks to `[0,1]`.
- [x] Compute `effective_mask = atlas_mask_slot * (0.25 + 0.75 * visible_slot)`.
- [x] Read `configs/magenta_policy.json` using schema `{ "default": false, "per_slot": { ... } }`.
- [x] For current slot, compute `magenta_enabled_for_slot = per_slot.get(slot_name, default)`.
- [x] If `magenta_enabled_for_slot` is true, set magenta special target class `1` where target RGB is exactly `[255, 0, 255]`.
- [x] If `magenta_enabled_for_slot` is false, set all special targets to class `0` and set special loss weight to `0` for that slot.
- [x] Set class `0` normal RGB elsewhere inside atlas mask.
- [x] Ignore special loss where atlas mask is `0`.

## 30. SlotNet loss

- [x] Implement `models/losses.py`.
- [x] Implement RGB L1 loss with effective mask.
- [x] Implement Sobel edge loss with effective mask.
- [x] Implement special magenta cross-entropy loss inside atlas mask only if magenta policy is enabled.
- [x] Do not use GAN loss in V0.
- [x] Do not use VGG/perceptual loss in default V0.
- [ ] Compute SSIM as a validation/debug metric if easy, but do not make it a blocking V0 training loss.
- [ ] Do not compute RGB/Sobel losses with a flat `.mean()` over the whole padded slot tensor.
- [x] Use mask-sum normalization for RGB: `L_rgb = sum(abs(pred_rgb - target_rgb) * effective_mask) / (sum(effective_mask) * 3 + 1e-8)`.
- [x] Use mask-sum normalization for Sobel: `L_sobel = sum(abs(sobel(pred_rgb) - sobel(target_rgb)) * effective_mask) / (sum(effective_mask) * sobel_channels + 1e-8)`.
- [x] Ensure a sparse slot with large padding produces the same gradient scale per active pixel as a dense slot.
- [x] Use `L_special = cross_entropy(special_logits, special_target)` only where atlas mask is active.
- [x] Use default `L_slot = slot_weight * (1.00 * L_rgb + 0.50 * L_sobel + 1.00 * L_special_magenta_if_enabled)`.
- [ ] If smoke outputs are noisy/over-sharpened, lower `--sobel-weight` to `0.25`.
- [ ] If smoke outputs are blurry, raise `--sobel-weight` to `1.00` before adding any perceptual/GAN loss.
- [ ] Do not train slots whose effective slot weight is zero.
- [ ] Ensure validation metric uses slot-weighted aggregate loss, not raw average.
- [ ] Track pressed-state RGB/Sobel metrics separately for `CBUTTONS`, `PLAYPAUS`, `SHUFREP`, `VOLUME`, `BALANCE`, and `POSBAR` visible regions.
- [ ] Track hidden-state RGB/Sobel metrics separately for the same state-bearing slots.
- [ ] Use numeric rare-state escalation trigger: if `pressed_state_rgb_mae > 1.5 * normal_state_rgb_mae` for any state-bearing slot on validation, regenerate data with `--variants 32 --state-balanced`.
- [ ] Also escalate if `hidden_state_rgb_mae > 1.5 * visible_current_state_rgb_mae` for any state-bearing slot.
- [ ] Do not add GAN/perceptual loss first when rare-state metrics fail; fix state coverage before changing model/loss family.

## 31. SlotNet grouped batching

- [ ] Do not batch different slot sizes together.
- [ ] Training loop chooses one active slot id for each step.
- [ ] Choose slot id with probability proportional to slot loss weight.
- [ ] Build an entire batch for that slot.
- [ ] Crop all samples in batch to that slot size.
- [ ] Forward SlotNet on that slot batch.
- [ ] Compute slot loss.
- [ ] Use batch size `8` for `MAIN`.
- [ ] Use batch size `8` for `TITLEBAR`.
- [ ] Use batch size `32` for `CBUTTONS`.
- [ ] Use batch size `24` for `SHUFREP`.
- [ ] Use batch size `32` for `MONOSTER`.
- [ ] Use batch size `32` for `PLAYPAUS`.
- [ ] Use batch size `6` for `EQMAIN`.
- [ ] Use batch size `8` for `PLEDIT`.
- [ ] Use batch size `32` for `POSBAR`.
- [ ] Use batch size `8` for `VOLUME`.
- [ ] Use batch size `8` for `BALANCE`.
- [ ] If OOM occurs, halve the batch size for the affected slot.

## 32. SlotNet image augmentation

- [ ] Apply the same random letterbox/pad augmentation as GeoNet with probability `p = 0.30`, updating rects before cropping.
- [ ] Apply photometric augmentation to the full input view before cropping.
- [ ] Do not apply arbitrary geometric augmentation in PyTorch beyond letterbox simulation.
- [ ] Geometry variation otherwise comes from Cranamp saved renders and SlotNet rect jitter.
- [ ] Brightness factor sampled from `[0.88, 1.12]`.
- [ ] Contrast factor sampled from `[0.88, 1.12]`.
- [ ] Saturation factor sampled from `[0.88, 1.12]`.
- [ ] Hue offset sampled from `[-0.03, 0.03]`.
- [ ] Gaussian noise probability `p = 0.25`.
- [ ] Gaussian noise sigma sampled from `[0.0, 0.015]`.
- [ ] Gaussian blur probability `p = 0.15`.
- [ ] Gaussian blur kernel size `3`.
- [ ] Gaussian blur sigma sampled from `[0.0, 0.8]`.
- [ ] JPEG roundtrip probability `p = 0.25`.
- [ ] JPEG quality sampled as integer from `[70, 95]`.

## 33. SlotNet training script

- [x] Implement `train_slotnet.py`.
- [x] Use AdamW optimizer.
- [x] Use Stage A learning rate `1e-4`.
- [ ] Use Stage B learning rate `5e-5`.
- [x] Use weight decay `1e-4`.
- [ ] Use cosine decay scheduler with 5% warmup.
- [ ] Use AMP/mixed precision.
- [x] Use gradient clipping with `max_norm = 1.0`.
- [ ] If a slot OOMs at its default batch size, halve micro-batch and set `--grad-accum-steps 2` before reducing effective batch.
- [ ] Save checkpoint every `5000` steps.
- [x] Save `last.safetensors`.
- [x] Save `best.safetensors`.
- [x] Save `config.yaml`.
- [x] Save `metrics.jsonl`.
- [ ] Save debug grids every `1000` training steps.
- [ ] Train Stage A with ground-truth rects plus jitter.
- [ ] Train Stage B with mixed ground-truth and frozen GeoNet-predicted rects plus jitter.
- [ ] Resume Stage B from Stage A best checkpoint.

Stage A command:

```bash
python train_slotnet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --atlas-profile configs/atlas_v1.json \
  --rect-source gt \
  --rect-jitter \
  --steps 150000 \
  --lr 1e-4 \
  --grad-accum-steps 1 \
  --amp \
  --out runs/slotnet_v1_stage_a
```

Stage B command:

```bash
python train_slotnet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --atlas-profile configs/atlas_v1.json \
  --geonet runs/geonet80_v0/best.safetensors \
  --rect-source mixed_gt_pred \
  --rect-mix-ratio 0.5 \
  --rect-jitter \
  --steps 100000 \
  --lr 5e-5 \
  --grad-accum-steps 1 \
  --amp \
  --resume runs/slotnet_v1_stage_a/best.safetensors \
  --out runs/slotnet_v1_stage_b
```

## 34. Debug outputs

- [ ] Write GeoNet debug images to `debug/geonet/`.
- [ ] GeoNet debug images must overlay predicted rects and target rects on input view.
- [ ] Write SlotNet debug images to `debug/slotnet/`.
- [ ] SlotNet debug image must include source crop.
- [ ] SlotNet debug image must include target slot.
- [ ] SlotNet debug image must include predicted slot.
- [ ] SlotNet debug image must include absolute diff.
- [ ] SlotNet debug image must include effective loss mask.
- [ ] Every validation epoch, run the full pipeline on 16 validation samples.
- [ ] Full pipeline debug must include original input view.
- [ ] Full pipeline debug must include GeoNet rect overlay.
- [ ] Full pipeline debug must include assembled predicted atlas.
- [ ] Full pipeline debug must include target atlas.
- [ ] Full pipeline debug must include Cranamp-rendered predicted skin using same `params.json`.
- [ ] Full pipeline debug must include target/input render.
- [ ] Full pipeline debug must include render diff.
- [ ] Use these render debug images to catch “atlas looks okay but rendered skin is bad” failures.

## 35. Inference/export script

- [ ] Implement `infer_skin.py`.
- [ ] Load arbitrary mockup input image.
- [ ] Letterbox image to `960x1728`.
- [ ] Run GeoNet80 backbone/FPN and detection heads.
- [ ] Decode rects `[80,5]` from GeoNet detection heads.
- [ ] Derive EQ band rects 30-39 from predicted `eq_sliders_group` using the shared deterministic function.
- [ ] Sample GeoNet state-anchor features from FPN `P` at predicted/derived rect centers using 3x3 average pooling.
- [ ] Decode state vector `[32]` from those sampled local features.
- [ ] For each active V0 slot, crop source region using predicted rects.
- [ ] Run SlotNet for each active V0 slot.
- [ ] Paste predicted RGB into the slot position in a `1024x1024` atlas.
- [ ] For ignored slots, paste default skin asset or blank/default placeholder.
- [ ] Apply special-class snapping after SlotNet prediction.
- [ ] For each slot, read `magenta_enabled_for_slot = per_slot.get(slot_name, default)` from `configs/magenta_policy.json`.
- [ ] If magenta is enabled for that slot and magenta special probability is greater than `0.60`, set RGB exactly to `#FF00FF`.
- [ ] If magenta is disabled for that slot, do not snap magenta; export RGB as predicted/postprocessed.
- [ ] Export BMP files using `configs/export_profile_classic.json`.
- [ ] Copy or generate default `TEXT.bmp`.
- [ ] Copy or generate default `NUMBERS.bmp`.
- [ ] Copy or generate `PLEDIT.TXT`.
- [ ] Copy or generate `VISCOLOR.TXT`.
- [ ] Zip output folder into `skin.wsz`.
- [ ] Save `atlas.png` for inspection.
- [ ] Save `rects_pred.f32` for inspection.
- [ ] Save `state_pred.f32` for inspection.

Command:

```bash
python infer_skin.py \
  --image mockup.png \
  --geonet runs/geonet80_v0/best.safetensors \
  --slotnet runs/slotnet_v1_stage_b/best.safetensors \
  --atlas-profile configs/atlas_v1.json \
  --export-profile configs/export_profile_classic.json \
  --default-skin assets/default_skin \
  --out out_skin
```

Expected output:

```text
out_skin/atlas.png
out_skin/MAIN.bmp
out_skin/EQMAIN.bmp
out_skin/PLEDIT.bmp
out_skin/CBUTTONS.bmp
out_skin/TITLEBAR.bmp
out_skin/SHUFREP.bmp
out_skin/VOLUME.bmp
out_skin/BALANCE.bmp
out_skin/MONOSTER.bmp
out_skin/POSBAR.bmp
out_skin/PLAYPAUS.bmp
out_skin/TEXT.bmp
out_skin/NUMBERS.bmp
out_skin/PLEDIT.TXT
out_skin/VISCOLOR.TXT
out_skin/skin.wsz
```

## 36. Atlas export helper

- [x] Implement `scripts/05_export_atlas_to_skin.py`.
- [x] Input is `atlas.png`.
- [x] Input is `configs/atlas_v1.json`.
- [x] Input is `configs/export_profile_classic.json`.
- [x] Input is `assets/default_skin/`.
- [x] Crop each active slot to exact export dimensions.
- [x] Save BMP files.
- [x] Fill ignored/default files from default skin.
- [x] Save text config files.
- [x] Zip final skin as `.wsz`.
- [x] Unit-test exported file sizes.
- [ ] Unit-test exported skin loads in Cranamp smoke test if CLI supports it.

## 37. Evaluation pipeline

- [x] Implement `eval_pipeline.py`.
- [ ] Run full pipeline on test split.
- [ ] Use GeoNet predicted rects, not ground-truth rects, in full-pipeline evaluation.
- [ ] Run SlotNet on predicted rects.
- [ ] Assemble atlas.
- [ ] Export BMPs.
- [ ] Render predicted skin using Cranamp and same `params.json` as the source input.
- [ ] Compare predicted render to target/input render.
- [ ] Save visual comparisons.
- [ ] Write metrics JSON/CSV.

Command:

```bash
python eval_pipeline.py \
  --test data_v0/test.csv \
  --geonet runs/geonet80_v0/best.safetensors \
  --slotnet runs/slotnet_v1_stage_b/best.safetensors \
  --atlas-profile configs/atlas_v1.json \
  --export-profile configs/export_profile_classic.json \
  --cranamp-cli ./cranamp_cli/cranamp-cli \
  --out eval/v0
```

Evaluation metrics:

- [ ] GeoNet mean IoU for visible rects.
- [ ] GeoNet median center error in pixels.
- [ ] GeoNet p95 center error in pixels.
- [ ] GeoNet main/eq/playlist IoU.
- [ ] GeoNet small-control IoU.
- [ ] GeoNet state L1.
- [ ] SlotNet slot-weighted RGB MAE.
- [ ] SlotNet slot-weighted Sobel MAE.
- [ ] SlotNet per-slot RGB MAE.
- [ ] SlotNet magenta pixel precision.
- [ ] SlotNet magenta pixel recall.
- [ ] SlotNet pressed-state region RGB MAE.
- [ ] SlotNet hidden-state region RGB MAE.
- [ ] SlotNet state-bearing slot muddiness report for `CBUTTONS`, `SHUFREP`, `PLAYPAUS`, `VOLUME`, `BALANCE`, and `POSBAR`.
- [ ] Pipeline predicted atlas valid size.
- [ ] Pipeline exported BMPs valid size.
- [ ] Pipeline Cranamp render succeeds.
- [ ] Pipeline render MAE against target/input render.
- [ ] Pipeline visual comparison grids.

V0 pass condition:

- [ ] 100% of test samples export BMPs without size errors; any export-size failure is a pipeline bug, not model tolerance.
- [ ] At least 95% of test samples render in Cranamp without crash.
- [ ] GeoNet main/eq/playlist mean IoU is greater than `0.85`.
- [ ] GeoNet median center error is less than `8 px`.
- [ ] Slot-weighted RGB MAE is less than `0.08` on normalized RGB.
- [ ] Manual visual inspection of 50 random held-out examples shows correct slot placement.
- [ ] The recovered `.wsz` should load in Cranamp.
- [ ] If classic Winamp 5.x is available, the recovered `.wsz` should also load there without obvious file-size errors.

## 38. Training run order

- [x] Run skin scan. (7787 .wsz in `skins_raw/`, ~7774 fingerprint-valid after dedup)
- [x] Dump Cranamp classic export profile.
- [ ] Run magenta verification and write `configs/magenta_policy.json`.
- [ ] Run Cranamp render-throughput benchmark and decide single-process vs parallel rendering.
- [x] Pack skins into atlases. (full pack: 7754 ok / 7 reject / 25 error from 7787 raw — 99.6% pack rate)
- [x] Render offline randomized dataset. (smoke + V0: 8000 samples = 500 skins × 16 variants; 0 failures on 24-worker pool)
- [x] Make train/val/test splits by skin. (V0: 6288 train / 848 val / 864 test, split by skin_id hash)
- [x] Run dataset checker. (V0 passed all 8000 samples; contact sheets at data_v0/debug/{contact.png,contact_v0.png})
- [x] Run Smoke training. (CUDA torch 2.11.0+cu128 installed into project .venv; RTX 2070 mobile active)
- [x] Treat Smoke as pipeline validation only; Smoke outputs are expected to look visually broken.
- [x] Smoke pass condition is end-to-end execution, shape correctness, export correctness, and loadable `.wsz`. (All four hold: pipeline runs, 25/25 tests pass, 16 skins exported, .wsz structure verified.)
- [x] Train GeoNet Smoke. (50 steps batch=1 on RTX 2070; total loss 928k → 1.18; runs/geonet80_smoke/)
- [x] Train SlotNet Smoke. (100 steps on MAIN slot; total loss 1.85 → 0.66; runs/slotnet_smoke/)
- [x] Run Smoke inference/export. (eval_pipeline.py exported 16 test skins, each with full 18-file .wsz zip; eval/smoke/)
- [ ] Verify exported BMPs load in Cranamp. (Needs an actual Cranamp/Winamp player; deferred to user.)
- [ ] Run V0 training on `500 skins x 16 variants`.
- [ ] Train GeoNet V0.
- [ ] Train SlotNet V0 Stage A.
- [ ] Train SlotNet V0 Stage B.
- [ ] Evaluate V0 pipeline.
- [ ] Manually inspect 50 examples.
- [ ] Only after V0 passes, try qualitative AI mockup inference.

Smoke commands:

```bash
python scripts/04_check_dataset.py --data data_v0
python train_geonet.py --train data_v0/train.csv --val data_v0/val.csv --batch 2 --epochs 2 --lr 2e-4 --amp --out runs/geonet80_smoke
python train_slotnet.py --train data_v0/train.csv --val data_v0/val.csv --atlas-profile configs/atlas_v1.json --rect-source gt --rect-jitter --steps 1000 --lr 1e-4 --amp --out runs/slotnet_smoke
python eval_pipeline.py --test data_v0/test.csv --geonet runs/geonet80_smoke/best.safetensors --slotnet runs/slotnet_smoke/best.safetensors --out eval/smoke
```

V0 commands:

```bash
python scripts/04_check_dataset.py --data data_v0

python train_geonet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --batch 4 \
  --grad-accum-steps 1 \
  --epochs 80 \
  --lr 2e-4 \
  --amp \
  --out runs/geonet80_v0

python train_slotnet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --atlas-profile configs/atlas_v1.json \
  --rect-source gt \
  --rect-jitter \
  --steps 150000 \
  --lr 1e-4 \
  --grad-accum-steps 1 \
  --amp \
  --out runs/slotnet_v1_stage_a

python train_slotnet.py \
  --train data_v0/train.csv \
  --val data_v0/val.csv \
  --atlas-profile configs/atlas_v1.json \
  --geonet runs/geonet80_v0/best.safetensors \
  --rect-source mixed_gt_pred \
  --rect-mix-ratio 0.5 \
  --rect-jitter \
  --steps 100000 \
  --lr 5e-5 \
  --grad-accum-steps 1 \
  --amp \
  --resume runs/slotnet_v1_stage_a/best.safetensors \
  --out runs/slotnet_v1_stage_b

python eval_pipeline.py \
  --test data_v0/test.csv \
  --geonet runs/geonet80_v0/best.safetensors \
  --slotnet runs/slotnet_v1_stage_b/best.safetensors \
  --out eval/v0
```

## 39. Checkpointing/logging

- [ ] Save GeoNet checkpoints in `runs/geonet80_v0/`.
- [ ] Save SlotNet Stage A checkpoints in `runs/slotnet_v1_stage_a/`.
- [ ] Save SlotNet Stage B checkpoints in `runs/slotnet_v1_stage_b/`.
- [ ] Every run directory must contain `last.safetensors`.
- [ ] Every run directory must contain `best.safetensors`.
- [ ] Every run directory must contain `config.yaml`.
- [ ] Every run directory must contain `metrics.jsonl`.
- [ ] GeoNet checkpoints every epoch.
- [ ] SlotNet checkpoints every 5000 steps.
- [ ] Log loss components separately.
- [ ] Log per-slot validation metrics.
- [ ] Log slot sampling counts.
- [ ] Log learning rate.
- [ ] Log GPU memory peak.
- [ ] Log random seed.

## 40. Determinism

- [ ] For debugging, set Python random seed.
- [ ] For debugging, set NumPy random seed.
- [ ] For debugging, set PyTorch random seed.
- [ ] For debugging, set CUDA random seed.
- [ ] For debugging, use `torch.use_deterministic_algorithms(True)`.
- [ ] For production training, deterministic algorithms may be disabled for speed.
- [ ] Even in production, set and log all seeds.
- [ ] Every dataset sample must be reproducible from `skin_id`, `variant_id`, `seed`, and `params.json`.
- [ ] Do not allow unlogged randomness in Cranamp renders.

Debug seed snippet:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
```

## 41. Unit tests

- [x] `test_atlas_pack.py` verifies BMP packing positions.
- [x] `test_atlas_pack.py` verifies atlas shape `1024x1024x3`.
- [x] `test_atlas_pack.py` verifies atlas mask values.
- [x] `test_rect_encoding.py` verifies normalized rect encoding and clipping.
- [x] `test_rect_encoding.py` verifies invisible rect zeroing.
- [x] `test_rect_encoding.py` verifies deterministic `derive_eq_band_rects` output for fixed inputs.
- [x] `test_rect_encoding.py` verifies EQ band derivation is byte-identical for training and inference helper calls.
- [x] `test_slot_crop.py` verifies grid_sample crop dimensions.
- [x] `test_slot_crop.py` verifies full-image crop behavior.
- [x] `test_slot_crop.py` verifies jittered rect clipping.
- [x] `test_slot_crop.py` verifies log-scale-ratio channel values for known rect/slot sizes.
- [x] `test_geonet_state.py` verifies state-anchor 3x3 average-pool sampling.
- [x] `test_geonet_state.py` verifies state-anchor jitter is active only during training.
- [x] `test_visible_atlas_mask.py` verifies u32 provenance ID encode/decode.
- [x] `test_magenta_policy.py` verifies per-slot magenta policy lookup and default fallback.
- [x] `test_state_regions.py` verifies state-region rectangles are inside slot capacity rectangles.
- [ ] `test_state_regions.py` verifies every non-empty state-bearing slot can compute normal/pressed/hidden metrics without guessing rectangles.
- [x] `test_mask_normalized_loss.py` verifies RGB/Sobel loss uses mask-sum normalization and is invariant to padding area.
- [x] `test_export_skin.py` verifies exported BMP dimensions match export profile.
- [x] `test_export_skin.py` verifies ignored/default files are created.
- [x] `test_export_skin.py` verifies `.wsz` zip is created.

## 42. GPU rental and local usage

- [ ] Use local RTX 2070 mobile only for skin packing, Cranamp CLI debugging, dataset checks, tiny smoke tests, and code validation.
- [ ] Do not plan real V0 training on RTX 2070 mobile.
- [ ] Rent a GPU for real training.
- [ ] Recommended first rental GPU: RTX 4090 24GB.
- [ ] Comfortable rental GPU: RTX A6000 / RTX 6000 Ada / other 48GB card.
- [ ] Use RunPod or Vast.ai as candidate rental providers.
- [ ] Prefer RunPod for first run if simpler environment matters.
- [ ] Prefer Vast.ai later if scripts are stable and lower cost matters.
- [ ] Minimum rental machine: RTX 4090 24GB, 16 vCPU, 64GB RAM, 300GB NVMe for V0.
- [ ] Comfortable rental machine: 48GB GPU, 24+ vCPU, 128GB RAM, 1TB NVMe.
- [ ] Do not use interruptible/spot rental until checkpoint resume is tested.
- [ ] Use Docker image or environment equivalent to PyTorch with CUDA.
- [ ] Sync dataset and runs carefully; do not lose checkpoints on rental shutdown.

Cloud setup commands:

```bash
git clone <repo>
cd cranamp-atlas-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected V0 rental rough scope:

```text
500 skins x 16 renders = 8,000 samples
GeoNet: several hours on RTX 4090-class GPU
SlotNet: 12-36 hours on RTX 4090-class GPU depending batch/IO
First serious V0: one 24-48 hour rental should be enough
```

## 43. Implementation order for the single coding agent

- [x] Step 1: Create repo structure and config stubs.
- [x] Step 2: Create `requirements.txt`.
- [x] Step 3: Create `configs/atlas_v1.json`.
- [x] Step 4: Create `configs/components_v1.json`.
- [x] Step 5: Create `configs/slot_sources_v1.json`.
- [x] Step 6: Create placeholder `configs/export_profile_classic.json`.
- [x] Step 7: Inspect user-provided Cranamp repo.
- [x] Step 8: Implement or wrap `cranamp-cli dump-classic-spec`.
- [x] Step 9: Implement or wrap `cranamp-cli render-random`.
- [x] Step 10: Implement or wrap `cranamp-cli render-with-params`.
- [x] Step 11: Ensure Cranamp CLI outputs `view.png`, `rects.f32`, `state.f32`, `visible_atlas_mask.png`, and `params.json`.
- [x] Step 12: Implement `scripts/00_scan_skins.py`.
- [x] Step 13: Implement `scripts/01_pack_skins.py`.
- [x] Step 14: Support `.wsz` and folder skins.
- [x] Step 15: Support case-insensitive filenames.
- [x] Step 16: Use default assets for missing optional BMPs.
- [x] Step 17: Write atlas, mask, slot weights, metadata, and valid skins CSV.
- [x] Step 18: Implement `scripts/02_render_dataset.py`.
- [x] Step 19: Implement deterministic seeds.
- [x] Step 20: Render saved offline dataset.
- [x] Step 21: Implement `scripts/03_make_splits.py`.
- [x] Step 22: Implement `scripts/04_check_dataset.py`.
- [x] Step 23: Implement debug contact sheets.
- [x] Step 24: Implement `models/geonet80.py`.
- [x] Step 25: Implement CenterNet label builder.
- [x] Step 26: Implement GeoNet losses.
- [x] Step 27: Implement `train_geonet.py`.
- [x] Step 28: Implement GeoNet inference rect decoding.
- [ ] Step 29: Implement GeoNet debug overlay.
- [x] Step 30: Implement `models/crop.py`.
- [x] Step 31: Implement `models/atlas.py`.
- [x] Step 32: Implement `models/losses.py`.
- [x] Step 33: Implement `models/slotnet_v1.py`.
- [x] Step 34: Implement SlotNet rect jitter.
- [ ] Step 35: Implement SlotNet grouped-by-slot batching.
- [ ] Step 36: Implement SlotNet Stage A training.
- [ ] Step 37: Implement SlotNet Stage B mixed GT/predicted rect training.
- [ ] Step 38: Implement SlotNet debug grids.
- [x] Step 39: Implement `scripts/05_export_atlas_to_skin.py`.
- [ ] Step 40: Implement `infer_skin.py`.
- [x] Step 41: Implement `eval_pipeline.py`.
- [x] Step 42: Implement unit tests.
- [ ] Step 43: Run Smoke dataset and training.
- [ ] Step 44: Fix Smoke failures.
- [ ] Step 45: Run V0 dataset and training.
- [ ] Step 46: Evaluate V0.
- [ ] Step 47: Manually inspect 50 V0 outputs.
- [ ] Step 48: Only after V0, try qualitative AI mockups.

## 44. Known V0 limitations

- [ ] V0 is trained on Cranamp-rendered synthetic geometry, not on AI creative mockups.
- [ ] V0 may fail on AI mockups whose visual distribution differs strongly from Cranamp renders.
- [ ] V0 cannot preserve creative details that strict classic Winamp slots cannot represent.
- [ ] V0 cannot preserve ten different EQ slider cats if the target format only has one shared classic slider asset.
- [ ] V0 does not train font bitmaps.
- [ ] V0 does not output a custom Cranamp manifest.
- [ ] V0 does not solve unpaired creative atlas generation.

## 45. Planned V1 after V0 passes

- [ ] Add identity-preserving local img2img data only after V0 passes.
- [ ] Identity-preserving prompts must preserve same skin, same colors, same buttons, same UI layout, and same visual identity.
- [ ] Target remains original atlas only when input preserves original atlas visual identity.
- [ ] Do not use prompts like “make it kitten-themed/cyberpunk/watercolor” paired to the original atlas.
- [ ] Add more photometric and renderer-distribution augmentations if GeoNet fails on AI mockups.
- [ ] Consider a differentiable PyTorch Cranamp-like compositor after V0, not before.
- [ ] Consider render loss `render(predicted_atlas) ≈ input_mockup` after V0, not before.

## 46. Planned V2 after V1

- [ ] Add unpaired creative render-loss adaptation for AI mockups.
- [ ] Add differentiable compositor for training/refinement.
- [ ] Add atlas plausibility prior if render-loss optimization produces broken sprite strips.
- [ ] Add side-by-side human evaluation UI: input mockup on left, rendered prediction on right, 1-5 rating, free comment.
- [ ] Use human evaluation to diagnose what automated metrics miss.

## 47. Planned V3 after V2

- [ ] Consider Cranamp extended skin format only after strict classic V0/V1/V2 evidence shows limitations.
- [ ] Possible extension: per-EQ-band thumb sprites.
- [ ] Possible extension: per-window decorative overlays.
- [ ] Possible extension: custom scrollbar/thumb sprites.
- [ ] Possible extension: richer titlebar/window variants.
- [ ] Possible extension: multiple state families beyond classic Winamp reuse rules.
- [ ] Add new atlas slots using reserved capacity or a new manifest version.
- [ ] Keep backward compatibility with classic export where possible.

## 48. Final V0 milestone definition

- [ ] Given a Cranamp randomized render of a held-out real skin, the system produces a packed atlas.
- [ ] The packed atlas exports to BMPs with correct dimensions.
- [ ] The exported skin folder loads in Cranamp.
- [ ] The exported `.wsz` loads in Cranamp.
- [ ] If classic Winamp 5.x is available, the exported `.wsz` loads there without file-size errors.
- [ ] Rendered exported skin visually resembles the source skin render.
- [ ] Controls appear in correct atlas slots.
- [ ] SlotNet survives imperfect GeoNet rects through Stage B mixed training.
- [ ] V0 evaluation metrics meet pass thresholds.
- [ ] Only after this milestone should the user spend time on creative AI mockup conversion experiments.
