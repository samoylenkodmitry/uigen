# SlotNet V7 Unified Plan

Date: 2026-05-23

## Goal

Build one deployed model:

```text
input mockup/render image -> exported BMP tensors -> .wsz
```

The model should have two internal functions:

```text
Observer/copy:
  preserve visible source evidence from the input image

Completer:
  infer hidden sprite states and complete full exported BMP files
```

This must remain one checkpoint and one inference call. Staged training is
allowed; two deployed models are not the goal.

## Why V7

V5 proved the exported-BMP contract and identity recognition:

```text
V5 Gate C:
  retrieval top1 = 1.000
  MAE 0.01848
  hit5 0.790
```

But V5 still generated high-frequency pixels from compressed features and
softened hard photographic/textured skins.

V6 proved source correspondence:

```text
best UV median = 0.63 px
```

But V6 copy/refine did not clear visible copy quality:

```text
best visible_mae = 0.0185
target < 0.01
```

So V7 should combine V6's source-preserving correspondence with a learned
completion/inpainting prior over exported skin assets.

## Architecture Sketch

### Shared Encoder

Input:

```text
view [B, 3, 1728, 960]
```

Use a multi-scale CNN/FPN-style encoder:

```text
E1 [B, C,     1728, 960]  stride 1
E2 [B, 2C,     864, 480]  stride 2
E3 [B, 4C,     432, 240]  stride 4
E4 [B, 6C,     216, 120]  stride 8
E5 [B, 8C,     108, 60]   stride 16
```

Start:

```text
C = 24
style = MLP(mean_pool(E5))  # 192 or 256 dim
```

V7 should not rely only on E5 tokens. Use multi-scale evidence from E3/E4/E5
for local detail.

### Per-File Query Grid

For every trainable exported BMP:

```text
MAIN.bmp
EQMAIN.bmp
PLEDIT.bmp
CBUTTONS.bmp
SHUFREP.bmp
VOLUME.bmp
BALANCE.bmp
POSBAR.bmp
MONOSTER.bmp
PLAYPAUS.bmp
TITLEBAR.bmp
```

Create query grids at exact file scale:

```text
query_h = ceil(file_h / divisor)
query_w = ceil(file_w / divisor)
```

Suggested divisors:

```text
hard / dense files: divisor = 2 or 4
  MAIN, EQMAIN, PLEDIT, VOLUME, BALANCE, CBUTTONS

simple files: divisor = 4
```

Each query token gets:

```text
file embedding
output-space Fourier x/y coordinates
global style
multi-scale local evidence from encoder features
```

Prefer sparse/deformable multi-scale sampling over dense attention to every
high-resolution token.

## Observer / Copy Branch

Per file, predict:

```text
uv_grid           [B, 2, H, W]
copy_conf_logits  [B, 1, H, W]
patch_kernel      [B, K, H, W]
copy_residual     [B, 3, H, W]
```

V6 sampled one point:

```text
copy_rgb = grid_sample(view, uv_grid)
```

V7 should test a local patch sampler because V6's median UV was good but
visible copy quality still failed.

Patch sampler:

```text
K = 9 initially
offsets = 3x3 around uv in input-pixel units:
  (-1,-1), (0,-1), (1,-1),
  (-1, 0), (0, 0), (1, 0),
  (-1, 1), (0, 1), (1, 1)

patch_rgb[k] = grid_sample(view, uv_grid + offset_k)
weights = softmax(patch_kernel, dim=K)
copy_rgb = sum_k weights[k] * patch_rgb[k]
copy_refined = clamp(copy_rgb + residual_scale * tanh(copy_residual), 0, 1)
```

Start:

```text
residual_scale = 0.10 or 0.20
```

Track both bilinear and nearest diagnostics, but train the production copy
path consistently.

## Completer Branch

The completer receives per-file partial evidence:

```text
copy_rgb / copy_refined
copy_conf
support_mask
file embedding
Fourier coordinates
global style
optional local decoder features
```

It outputs:

```text
hidden_rgb_logits [B, 3, H, W]
blend_logits      [B, 1, H, W]
```

Final composition:

```text
copy_alpha = sigmoid(copy_conf_logits)
blend = sigmoid(blend_logits)
alpha = max(copy_alpha, blend)  # initial option; revisit after diagnostics
final_rgb = alpha * copy_refined + (1 - alpha) * sigmoid(hidden_rgb_logits)
```

The completer should be a small masked/gated U-Net per file group, not a
1x1 head. Use nearest upsampling.

For tiny files, use a shallow conv stack instead of heavy downsampling.

## State-Family Metadata

Do not leave hidden state structure implicit. Add a config after auditing
Cranamp/export code.

Target file:

```text
configs/state_families_classic.yaml
```

Initial contents should describe at least:

```text
VOLUME.bmp:
  type: vertical_strip
  frame_axis: y
  frame_count: <audit>
  frame_w: <audit>
  frame_h: <audit>

BALANCE.bmp:
  type: vertical_strip
  frame_axis: y
  frame_count: <audit>
  frame_w: <audit>
  frame_h: <audit>

CBUTTONS.bmp:
  type: button_state_sheet
  groups: <audit>

SHUFREP.bmp:
  type: toggle_state_sheet
  groups: <audit>

POSBAR.bmp:
  type: seekbar_sheet
  groups: <audit>
```

The first task is to inspect the actual Cranamp rendering/export semantics and
fill this config defensibly.

## Asset-Completion Training

Train the completer directly on exported BMP files with synthetic masks.

Input:

```text
observed_rgb = target_rgb * observed_mask
observed_mask
file id / coords / style
```

Target:

```text
full clean target_rgb
```

Mask families:

```text
40% true render masks from final-frame provenance
35% state-family masks: reveal one frame/state, hide siblings
20% random rectangle / stripe masks
 5% whole-file dropout
```

Whole-file dropout should stay rare to avoid mean-skin behavior.

This is not a second deployed model. It is a training mode for the same
completer branch used inside SlotNetV7Unified.

## Losses

For connected V7:

```text
L_final_l1       final_rgb vs clean target on support pixels
L_final_sobel    edge loss on final_rgb vs clean target
L_copy_rgb       visible pixels, copy_refined vs clean target
L_uv             visible pixels, uv_grid vs uv_target in pixel space
L_uv_tail        top-percent UV error, because V6 median passed while tails remained
L_conf           BCE(copy_conf_logits, visible_target), class balanced
L_hidden         hidden pixels, final_rgb vs clean target
```

Initial weights:

```text
1.00 * L_final_l1
1.50 * L_final_sobel
1.00 * L_copy_rgb
0.20 * L_uv
0.10 * L_uv_tail
0.25 * L_conf
1.00 * L_hidden
```

Tune only after one-skin gates expose a specific failure.

## Training Schedule

### Phase 0: Asset-Completion Pretraining

Train only completion branch and shared file embeddings on partial BMP masks.

Pass gates:

```text
one skin:
  MAE < 0.005
  hit5 > 0.95

16 skins:
  retrieval top1 = 1.0
  median MAE < 0.015
  hit5 > 0.90
```

### Phase 1: Observer / Copy Pretraining

Train render -> UV/copy/patch-copy using final-frame provenance.

Pass gates:

```text
visible_mae < 0.01
copy_conf_auc > 0.98
UV p50 < 1 px
UV p90 < 3 px
UV p95 < 5 px
```

### Phase 2: Connected Training With Teacher Forcing

Feed the completer a scheduled mix:

```text
80% oracle observed partials / 20% predicted observed partials
50% oracle / 50% predicted
20% oracle / 80% predicted
0% oracle / 100% predicted
```

This prevents the completer from only working on perfect labels.

### Phase 3: End-to-End Fine-Tune

Full path:

```text
view -> observer/copy -> completer -> final BMPs
```

Use lower LR for the pretrained completer than for the observer/file heads.

## Acceptance Gates

Do not run full corpus before these pass.

```text
Gate A: asset-completion one-skin
  MAE < 0.005
  hit5 > 0.95

Gate B: asset-completion 16-skin
  retrieval top1 = 1.0
  median MAE < 0.015
  hit5 > 0.90

Gate C: observer-copy one-skin
  visible MAE < 0.01
  AUC > 0.98
  UV p50 < 1 px
  UV p90 < 3 px
  UV p95 < 5 px

Gate D: connected one-skin
  final exported MAE < 0.01
  hit5 > 0.90

Gate E: hard-skin mini-gate
  tvxq
  zelda
  a_halo_so_bright_it_bleeds
  dragonzv30amp
  blair_razor_project
```

Gate E should compare against V5/V6 on:

```text
MAIN
EQMAIN
PLEDIT
VOLUME
BALANCE
CBUTTONS
```

## Immediate Claude Task

Do not implement the full V7 model first.

Start with:

```text
1. Audit state-sheet geometry in Cranamp/export code.
2. Add configs/state_families_classic.yaml.
3. Add tests that validate state-family rectangles stay inside TRAINABLE_EXPORT_SPECS.
4. Implement asset-completion mask generation for exported BMP tensors.
5. Add oracle/learnable tensor tests proving the completion loss/evaluator can reach near-zero.
```

Only after this scaffolding is correct should SlotNetV7Unified be implemented.

