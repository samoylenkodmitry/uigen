# V7 Attempt Report

Date: 2026-05-25

## Scope

This report summarizes the current V7 asset-completion attempt before asking
for an external review. It covers Phase 0 only:

```text
partial exported BMP evidence + observed mask + file/skin conditioning
-> completed exported BMP tensor
```

This is not the final deployed model. The final goal remains one inference
call:

```text
input mockup/render image -> exported BMP tensors -> .wsz
```

V7 Phase 0 is meant to prove that the internal completer can reconstruct
hidden sprite states once the observer/copy branch supplies partial evidence.

## Current V7 Design

The current completer is `V7Completer`:

- input: `observed_rgb`, `observed_mask`, `file_id`, optional `skin_id`
- output: completed RGB BMP for one trainable exported file
- model: small masked U-Net with file embedding, optional skin embedding, and
  Fourier coordinates
- final composition uses a hard copy path:

```python
final_rgb = observed_mask * observed_rgb + (1 - observed_mask) * generated_rgb
```

This hard-copy path was required. The earlier no-skip completer failed even
when given the complete target as input, because it had to learn identity
through the whole U-Net.

Training and evaluation operate only on Cranamp-supported exported BMP pixels.
Unsupported/default files are not part of the objective.

## What Worked

### Gate A: One-Skin Asset Completion

Gate A eventually passed on one skin after these fixes:

- hard-copy observed pixels
- same-file batching
- weighted file sampling
- Sobel/edge loss
- support-mask-only metrics
- skin/file exported BMP contract

Final Gate A result:

```text
aggregate supported_mae: 0.0029  target < 0.005
aggregate hit5:          0.960   target > 0.95
per-file supported_mae:  11/11 below 0.005
per-file hit5 floor:     no file below 0.90
```

This proves the one-skin completion path is representationally capable.

### Isolated Hard-File Probes

BALANCE/VOLUME isolated training passed strongly:

```text
BALANCE.bmp: 0.00432 mae, 0.968 hit5
VOLUME.bmp:  0.00533 mae, 0.941 hit5
```

EQMAIN-only probes also passed:

```text
whole_file EQMAIN:    0.00081 mae, 0.9996 hit5
state_family EQMAIN:  0.0024  mae, 0.981  hit5
```

So the model can memorize dense strip/state assets when they receive enough
gradient. Early multi-file failures were not proof that those files are
impossible.

## What Failed

### Gate B: Multi-Skin Asset Completion

The 14-skin Gate B asset-completion runs still fail the target:

```text
target mixed supported_mae < 0.015
target mixed hit5          > 0.90
```

Important runs:

```text
No skin embedding:
  supported_mae 0.069
  hit5          0.730

With skin embedding:
  supported_mae 0.043
  hit5          0.792

c64 from scratch:
  mixed mae     0.0396
  mixed hit5    0.849

staged c48 curriculum, best Phase B:
  mixed mae     0.0271
  mixed hit5    0.876
  state_family  0.0243 mae / 0.888 hit5

staged c48 curriculum, Phase C final:
  mixed mae     0.0282
  mixed hit5    0.864
  state_family  0.0276 mae / 0.861 hit5
```

Phase C regressed from Phase B. Adding random-rect/provenance after
state-family consolidation diluted the state-family fit; random-rect was
already easy and did not buy enough.

### Current Bottleneck After Curriculum

The bottleneck moved over time:

- BALANCE/VOLUME were initially bad, then targeted curriculum greatly improved
  them.
- POSBAR became the worst state-family file in the final curriculum report:

```text
POSBAR state_family: 0.0596 mae, 0.65 hit5
EQMAIN state_family: about 0.045 mae
MONOSTER state_family: about 0.038 mae
```

Worst skins after curriculum include photographic/high-frequency or saturated
skins such as `tvxq` and `a_halo`.

## Code Review Findings Before More Training

### 1. Weighted-Sampler Mask RNG Bug

I found and fixed a trainer bug in `train_v7_completer.py`.

Before the fix, weighted mode used a DataLoader, then called:

```python
dataset.set_epoch(step)
```

inside the training loop after the batch had already been fetched. That means
the current batch did not use the intended per-step mask RNG. With DataLoader
workers, later `set_epoch()` calls would not reliably affect worker dataset
copies either.

Fix:

- weighted mode now directly collates sampler indices
- `dataset.set_epoch(step)` happens before `dataset[index]`
- `--num-workers` is ignored for weighted mode with a log message

Validation:

```text
265 tests passed
```

This does not invalidate all previous results, but it makes them less clean as
evidence about the exact curriculum.

### 2. POSBAR State-Family Mask Looks Semantically Wrong

`configs/state_families_classic.yaml` currently defines POSBAR as:

```yaml
POSBAR.bmp:
  families:
    - name: parts
      kind: sprite
      rects:
        - {name: track, x: 0,   y: 0, w: 248, h: 10}
        - {name: thumb, x: 248, y: 0, w: 29,  h: 10}
```

The generic state-family mask reveals one rectangle and hides its siblings.
For POSBAR, that means it asks the model to reconstruct:

```text
full track from only the tiny thumb
or
thumb from only the track
```

But track and thumb are not alternate hidden states. They are complementary
parts of the same file. This is likely an artificial training/eval task, not a
real Cranamp state-completion task.

This matters because POSBAR is now one of the reported bottlenecks. The
bottleneck may be partly caused by an invalid mask recipe.

## Current Interpretation

Do not treat the latest Gate B failure as a clean architecture limit yet.

The strongest supported conclusions are:

1. The exported-BMP contract is correct.
2. One-skin completion works.
3. Hard strip assets are learnable in isolation.
4. Skin embedding is necessary for multi-skin completion.
5. File/curriculum weighting matters a lot.
6. More raw capacity alone did not solve Gate B.
7. At least one training-loop bug and one mask-semantics issue were found late.

Therefore more multi-hour Kaggle runs should stop until the mask/eval contract
is cleaned up.

## Questions For External Review

The main questions to ask:

```text
1. Should POSBAR be excluded from state_family masks/eval, or receive a
   custom POSBAR-specific mask?

2. For files that do not contain true sibling states, should state_family
   fallback to all-observed and rely on random_rect/provenance for completion?

3. Is the Gate B target meaningful if the state_family eval includes artificial
   tasks like reconstructing POSBAR track from thumb-only evidence?

4. After fixing weighted-mask RNG and POSBAR semantics, should we re-evaluate
   existing Phase B checkpoints before training again?

5. If Gate B still fails after corrected eval, should the next step be:
   - better mask curriculum,
   - per-file completer heads,
   - stronger per-skin conditioning,
   - or returning to the unified observer+completer end-to-end design?
```

## Recommended Next Steps

Before any new long training:

1. Keep the weighted-sampler mask RNG fix.
2. Fix POSBAR state-family semantics.
3. Add a regression test proving POSBAR state-family no longer hides track and
   thumb as siblings.
4. Re-evaluate the existing Phase B checkpoint under corrected masks.
5. Only then decide whether another Kaggle run is justified.

If another run is needed, it should be a short validation run first, not a new
5-hour probe.
