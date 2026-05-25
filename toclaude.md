# Message For Claude: Clean V7 Phase 0 Before Any More Training

Codex reviewed the latest GPT-5.5 Pro feedback against the current repo.
I agree with the core correction:

```text
Do not run another long Gate B training job yet.
The current V7 Phase 0 failure is still contaminated by task-definition,
metric, and mask-semantics issues.
```

Current baseline report:

```text
v7_attempt_report.md
```

Latest pushed commit:

```text
ffdae89 Document V7 attempt and fix weighted mask seeding
```

That commit also fixes the weighted-sampler mask RNG bug:

```text
weighted mode now sets dataset.set_epoch(step) before __getitem__
```

## Current Interpretation

Do not claim that V7 completer architecture has cleanly failed yet.

What is proven:

```text
1. One-skin V7 completer can pass.
2. Multi-skin V7 completer still fails.
3. Hard files can pass in isolation when they receive enough gradient.
4. Skin embedding helps, but it is oracle conditioning, not deployable.
5. At least one previous training-loop bug was found late.
6. POSBAR state_family semantics are likely wrong.
7. Current loss/eval denominator is not hidden-normalized.
```

Therefore the next work is cleanup and re-eval, not more Kaggle training.

## Stop Conditions

Do **not** launch another long local/Kaggle run until all of this is done:

```text
1. hidden-normalized losses/metrics implemented
2. state-family mask roles implemented
3. POSBAR no longer treated as track-vs-thumb sibling alternatives
4. tests prove the semantics
5. existing Phase B checkpoint re-evaluated under corrected metrics
```

## Task 1: Hidden-Normalized Losses And Metrics

Current V7 loss is support-normalized:

```python
support_masked_l1_loss(final_rgb, target_rgb, support_mask)
```

Because the model hard-copies observed pixels:

```python
final_rgb = observed_mask * observed_rgb + (1 - observed_mask) * generated_rgb
```

the numerator only has error on hidden pixels, but the denominator is still all
supported pixels. Mostly-observed samples therefore get diluted loss.

Add hidden-normalized functions to `models/losses_v7.py`.

Definitions:

```python
support = support_mask
observed = observed_mask * support
hidden = (1 - observed_mask) * support
```

Primary train/eval metrics:

```text
hidden_supported_mae
hidden_hit5
hidden_sobel_mae
observed_passthrough_mae
full_supported_mae   # debug/secondary only
full_supported_hit5  # debug/secondary only
```

Training loss should become:

```text
L = 1.00 * hidden_l1
  + 1.00 or 1.50 * hidden_sobel
  + 0.05 * full_supported_l1
```

If `hidden.sum() == 0`, skip that sample for the primary hidden loss or count
it only as a passthrough diagnostic.

Required tests:

```text
1. all-observed mask has zero hidden denominator and does not dilute loss
2. half-hidden sample normalizes by hidden pixels only
3. hard-copied observed pixels do not make hidden_mae look better
4. hidden_hit5 denominator is hidden pixel count, not support count
5. per-item hidden metrics work for same-file batches
```

## Task 2: Add `mask_role` To State Families

Current `state_family` mask logic hides siblings whenever a family has two or
more rectangles. That is only correct for true alternatives.

Extend `configs/state_families_classic.yaml` and `atlas_ai/state_families.py`
with:

```yaml
mask_role: alternatives | components | single
```

Rules:

```text
alternatives:
  reveal one rect/frame, hide sibling rects/frames

components:
  these rects are complementary parts, not alternate states
  state_family masking must not hide siblings

single:
  no state-family hiding
```

`StateRect` should carry `mask_role` so `atlas_ai/v7_masks.py` can filter.

`make_state_family_mask()` must only sample families where:

```python
mask_role == "alternatives"
```

If no alternatives exist for a file, state_family should fall back to
all-observed or be redistributed explicitly. Make the behavior visible in logs
and tests.

## Task 3: Correct Classic Roles

Use these initial role assignments.

True alternatives:

```text
CBUTTONS.bmp:
  each button pressed/unpressed family

SHUFREP.bmp:
  repeat on/off
  shuffle on/off
  eq_toggle on/off

MONOSTER.bmp:
  mono/stereo indicators

PLAYPAUS.bmp:
  status frames

VOLUME.bmp:
  slider_frames

BALANCE.bmp:
  slider_frames

EQMAIN.bmp:
  slider_frames
  on_button off/on
  auto_button off/on
```

Components / not state alternatives:

```text
MAIN.bmp:
  panel

TITLEBAR.bmp:
  title_strip
  corner_buttons

PLEDIT.bmp:
  title_bar
  side_borders
  footer
  scrollbar_thumb

POSBAR.bmp:
  track
  thumb

VOLUME.bmp:
  thumb

BALANCE.bmp:
  thumb

EQMAIN.bmp:
  chrome
  presets_label
  slider_thumb
```

The important one:

```text
POSBAR track and thumb are NOT sibling states.
Do not train/evaluate "track from thumb only" as a state-family task.
```

## Task 4: Regression Tests For Mask Semantics

Add tests before any training.

Required cases:

```text
POSBAR state_family must not hide track from thumb or thumb from track.
PLEDIT footer_left/footer_right must not be treated as alternatives.
EQMAIN chrome pieces must not be treated as alternatives.
TITLEBAR corner buttons must not be treated as alternatives.
CBUTTONS play pressed/unpressed must still hide one state from the other.
VOLUME/BALANCE slider_frames must still reveal one frame and hide siblings.
EQMAIN slider_frames must still reveal one of 28 and hide the other 27.
```

## Task 5: Re-Evaluate Existing Checkpoints First

After Tasks 1-4:

```text
Do not train first.
```

Re-evaluate the existing best Phase B checkpoint under corrected masks and
hidden-normalized metrics.

Report:

```text
full_supported_mae / hit5
hidden_supported_mae / hit5
per-file hidden metrics
per-skin hidden metrics
per-mode hidden metrics
observed_passthrough_mae
```

The point is to answer:

```text
Was the reported Gate B failure partly caused by invalid POSBAR masks and
support-normalized denominators?
```

## Task 6: Only Then Run A Short Sanity Train

If re-eval still fails, run a short corrected Gate B sanity train:

```text
2k-5k steps
oracle skin embedding ON
hidden-normalized loss
corrected mask roles
no long Kaggle run
```

Continue only if hidden metrics move in the expected direction.

## Longer-Term Validation Ladder

After the cleanup:

```text
Gate 0: direct hidden tensor optimization
  proves loss/eval code can reach near-zero

Gate A: one-skin completer
  hidden_supported_mae < 0.005-0.01
  hidden_hit5 > 0.95

Gate B: 14/16-skin oracle completer
  oracle skin_id allowed
  hidden_supported_mae < 0.015
  hidden_hit5 > 0.90

Gate C: replace oracle skin_id with deployable context encoder
  observed files/masks -> z_skin

Gate D: observer/copy branch
  render image -> observed BMP evidence

Gate E: unified end-to-end
  image -> observer -> context/completer -> BMPs
```

## What Not To Do

```text
Do not run another 5-hour probe to test a semantic question.
Do not train before re-evaluating existing checkpoints.
Do not treat skin_id embedding as final deployable conditioning.
Do not decide c48/c64 architecture from contaminated metrics.
Do not reintroduce full-atlas metrics, priors, unsupported files, or
distortion metadata side channels.
```

## Expected Deliverable

First deliverable should be code/tests only:

```text
1. hidden-normalized loss/metric functions
2. mask_role schema + loader support
3. corrected classic roles
4. mask-semantics regression tests
5. eval script updated to report hidden metrics
6. no training launched
```

Then pause for review.
