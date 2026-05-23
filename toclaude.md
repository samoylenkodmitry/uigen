# Message For Claude: Stop V6 Tuning, Plan V7 Unified Copy + Complete

Codex reviewed the GPT-5.5 Pro feedback and agrees with the core direction.

Do **not** continue same-recipe V6 Stage 2 tuning. Do **not** scale current V6
to multi-skin or rented GPU. V6 proved the correspondence mechanism, but the
current copy/refine architecture did not clear copy quality.

Current final V6 conclusions are in:

```text
v6_conclusions.md
REPORT_V6_STAGE2_DEFAULT_SKIN.md
```

Best V6 checkpoint:

```text
runs/slotnet_v6_default_skin_stage2_p4residual/snapshot_step005000.safetensors
```

Best V6 metrics:

```text
uv_median_px   0.63    target < 2.0    PASS
copy_conf_auc  0.9707  target > 0.98   near miss
visible_mae    0.0185  target < 0.01   PARTIAL / FAIL
```

## Decision

Move to V7 design. V7 should be one deployed model, one checkpoint, one
inference call:

```text
input image -> SlotNetV7Unified -> exported BMP tensors -> .wsz
```

But internally it needs two functions:

```text
1. Observer/copy: locate and preserve visible source evidence.
2. Completer: synthesize hidden sprite states using a learned skin prior.
```

This is not two production models. Staged training and auxiliary losses are
allowed, but inference must remain one graph.

## Do Not Do

```text
Do not build another pure generator.
Do not build a pure V6 copy-only model.
Do not rely on render variants to expose all hidden states.
Do not reintroduce prior/default atlas inputs.
Do not use distortion JSON side channels.
Do not use padded full-atlas loss/metrics as acceptance.
Do not scale V6 as-is.
```

## Immediate Next Work

Before implementing a full V7 model, prepare the design scaffolding:

1. Read `PLAN_V7_UNIFIED.md`.
2. Audit Cranamp state-sheet structure for the trainable exported BMPs.
3. Add a state-family metadata config for files like:

```text
VOLUME.bmp
BALANCE.bmp
CBUTTONS.bmp
SHUFREP.bmp
POSBAR.bmp
```

4. Build the asset-completion dataset/mask generator. This is the next useful
   code task.

The completion dataset should train:

```text
partial exported BMP evidence + observed mask + coords + file id
-> full clean exported BMP
```

Mask recipes:

```text
40% true render masks from final-frame provenance
35% state-family masks: reveal one state/frame, hide sibling states
20% random rectangle/stripe masks
 5% whole-file dropout
```

Do not start a long V7 training run until asset-completion and observer-copy
small gates are implemented and passing.

## V7 Gates

Use these as the first acceptance sequence:

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

Focus hard-skin scoring on:

```text
MAIN
EQMAIN
PLEDIT
VOLUME
BALANCE
CBUTTONS
```

## Current Recommendation

Start with the state-family metadata and asset-completion scaffolding. That is
the most direct way to test the part V6 cannot solve: hidden sprite states.

