# Message For Claude: Implement SlotNet V5

GPT-5.5 Pro agrees with the V4 conclusion. V4 is not a dead end; it gave us a
clean diagnosis.

## V4 Result To Preserve

V4 Gate 3:

```text
retrieval top1                  1.000
median true exported MAE        0.03879
target median exported MAE      < 0.02
```

Interpretation:

```text
V4 knows which skin it is seeing.
V4 cannot reconstruct enough skin-specific texture/detail.
```

The worst files were exactly the ones that need local texture/detail:

```text
EQMAIN    0.07830
VOLUME    0.05936
MAIN      0.05675
CBUTTONS  0.05161
BALANCE   0.05002
PLEDIT    0.05000
```

Do not continue V4 tuning. Do not rent GPU for V4. Do not retune file weights
as the main fix.

## Keep The Correct Contract

Keep all V4/V3.5 corrections:

- input render -> exact exported BMP tensors
- `TRAINABLE_EXPORT_SPECS`
- `exported_files_loss`
- static Cranamp-supported-pixel masks
- per-file MAE / hit5 / sobel metrics
- retrieval eval
- no prior/default atlas
- no distortion metadata
- no dynamic masks as model input
- no padded full-atlas pass/fail

Do not reintroduce any of the historical atlas/prior/distortion complexity.

## What Changes In V5

Current V35/V4:

```text
input render
-> CNN encoder
-> global mean-pooled style vector
-> per-file heads
-> exported BMP tensors
```

V5:

```text
input render
-> CNN encoder with spatial feature map
-> per-file learned query grids cross-attend to encoder spatial tokens
-> per-file decoders
-> exported BMP tensors
```

The file heads must receive local spatial evidence from the input render.
The global style vector can stay, but it must no longer be the only
conditioning path.

## Implementation Scope

Add:

```text
models/slotnet_v5.py
tests/test_slotnet_v5.py
```

Update:

```text
train_slotnet.py
infer_skin.py
scripts/09_eval_slotnet_overfit.py
scripts/11_eval_slotnet_retrieval.py
```

Required CLI support:

```text
--model-version 35|50
```

Default can remain `35` if safer, but V5 training commands must explicitly use:

```text
--model-version 50
```

Checkpoint detection must support:

```text
slotnet_version = 50
```

Do not break existing V35 checkpoints.

## Suggested V5 Architecture

Use a CNN encoder similar to V35, but expose the final spatial feature map
before global pooling.

For the current 960x1728 render inputs, expected rough shapes with
`base_channels=24`:

```text
enc1: [B,  24, 1728, 960]
enc2: [B,  48,  864, 480]
enc3: [B,  96,  432, 240]
enc4: [B, 144,  216, 120]
enc5: [B, 192,  108,  60]
```

Then:

```text
feature_map  = Conv1x1(enc5 -> attn_dim)       # [B, attn_dim, 108, 60]
tokens       = flatten(feature_map)            # [B, 6480, attn_dim]
tokens      += 2D positional encoding
global_style = MLP(mean(enc5))                 # [B, style_dim]
```

Start with:

```text
attn_dim = 128
attention_heads = 4
cross_attention_layers = 1
base_channels = 24
style_dim = 192
head_channels = 96
```

If memory is fine after smoke tests, a later variant can try:

```text
cross_attention_layers = 2
attn_dim = 160 or 192
```

Do not start with the larger version.

## Per-File Attention Head

For each `ExportFileSpec` with target size `H x W`, create a low-res query
grid:

```text
h0 = ceil(H / 8)
w0 = ceil(W / 8)
```

Each query grid cell should include:

```text
Fourier x/y coordinates
file embedding
global style conditioning
```

Then cross-attend:

```text
queries: file output grid tokens [B, h0*w0, attn_dim]
keys:    encoder spatial tokens  [B, 6480, attn_dim]
values:  encoder spatial tokens  [B, 6480, attn_dim]
```

After attention:

```text
reshape -> [B, attn_dim, h0, w0]
nearest-upsample conv decoder -> [B, 3, H, W]
```

Use nearest upsampling, not bilinear. Final resize/crop to exact `H x W` is
acceptable.

The output API must match V35:

```python
return {"files": {spec.file_name: logits}}
```

That keeps `exported_files_loss`, eval, and infer mostly unchanged.

## Attention Debugging

Add an optional attention debug dump. It does not need to be polished for the
first commit, but V5 should expose enough data to save heatmaps.

Goal:

```text
MAIN attention over input render
EQMAIN attention over input render
CBUTTONS attention over input render
PLEDIT attention over input render
VOLUME attention over input render
BALANCE attention over input render
```

Expected qualitative behavior:

```text
MAIN     -> main window/background/display regions
EQMAIN   -> equalizer window
CBUTTONS -> transport button row
PLEDIT   -> playlist area
VOLUME   -> volume slider region
BALANCE  -> balance slider region
```

Do not use attention maps as a loss yet. Use them only for debugging.

Suggested API:

```text
forward(view, return_attention=False)
```

When `return_attention=True`, include compact attention summaries. Avoid saving
full massive tensors by default.

## Tests Required Before Training

Add tests that run on CPU with tiny model settings:

```text
test V5 output files exactly match TRAINABLE_EXPORT_SPECS shapes
test slotnet_version buffer is 50
test forward returns {"files": ...}
test exported_files_loss backprop gives zero grad on unsupported pixels
test train/eval/infer model-version dispatch accepts 50 without breaking 35
```

Run full test suite before training:

```bash
.venv/bin/python -m pytest
```

## V5 Run Order

No GPU rental. Local short runs first.

### Gate A: One-Skin Overfit

Use an existing one-skin dataset, ideally BlueCurve or darkside.

```text
1 skin
32 variants
20k steps max
```

Pass:

```text
exported_pixels_mae < 0.01
exported_pixels_hit_5_255 > 0.90
```

V35/V4 already passed one-skin overfit, so V5 must pass. If it does not, the
implementation is wrong or the attention path is harming capacity.

### Gate B: Three-Skin Overfit

Pick three diverse skins:

```text
metallic/smooth
dark/textured
bright/light or pixel-art
```

Pass:

```text
retrieval top1 = 1.0
median exported MAE < 0.015-0.02
no visual identity collapse
```

### Gate C: Repeat V4 Gate 3

Use exactly:

```text
data_v4_16skin/train.csv
```

V4 baseline:

```text
retrieval top1                  1.000
median true exported MAE        0.03879
exported hit5                   0.668
```

V5 target:

```text
retrieval top1 >= 0.95
median true exported MAE < 0.02
```

At minimum, before calling V5 useful:

```text
median true exported MAE clearly below 0.03879
EQMAIN / MAIN / PLEDIT / VOLUME / CBUTTONS improve materially
```

If retrieval stays high and MAE drops, V5 is the right architecture.

## First Commit Should Be Code Only

First deliverable should be:

```text
SlotNetV5 implementation + dispatch + tests
```

Do not start long training until the tests pass and the model can do a single
forward/backward batch.

Recommended first smoke command after tests:

```bash
.venv/bin/python train_slotnet.py \
  --model-version 50 \
  --train data_v4_16skin/train.csv \
  --steps 2 \
  --batch 1 \
  --base-channels 8 \
  --style-dim 64 \
  --head-channels 32 \
  --out runs/slotnet_v5_smoke \
  --device cuda
```

Use smaller dimensions for smoke if needed; the real Gate A can return to the
recommended V5 defaults.

## Bottom Line

The data/loss/export contract is now good enough.

The next problem is architecture:

```text
global style bottleneck -> local spatial evidence path
```

Implement V5 local-attention per-file heads.
