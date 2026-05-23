# V7 Phase 0 / Gate B — 16-Skin Asset Completion

Status: **draft, awaiting review.** Do not launch until approved.

## Question

Does the V7 completer recipe that passed Gate A on one skin generalize to a
16-skin asset-completion task without architecture changes?

Gate A passed on `default_skin` at absolute step 55k with:

```text
aggregate supported_mae   0.0029
aggregate hit5            0.960
11/11 per-file sup_mae   <0.005
no per-file hit5 below 0.90
```

The recipe was: hard-mask copy contract, weighted same-file sampler with the
default V7 file weights, c48 capacity, sobel-weight 0.25, support-masked L1
loss. Architecture and losses are not changed in Gate B.

## Acceptance (initial)

Per the user's Gate B definition:

```text
median/aggregate supported_mae   < 0.015
hit5                             > 0.90
retrieval top1 = 1.0             only if a retrieval eval exists for completer outputs;
                                 currently the completer produces no global style vector,
                                 so retrieval is deferred to Gate D when the observer/copy
                                 path comes online.
```

Stronger ideal (mirroring Gate A):

```text
aggregate supported_mae          < 0.010
aggregate hit5                   > 0.95
no per-file hit5 below 0.85
```

Gate B passes on the initial thresholds; the stronger ideal is informational
for whether the next training pushes are worth doing.

## Data

Source: `data_v4_16skin/atlases/{skin_id}.{png,meta.json}` (16 skins) used by
V5 Gate C and V6 Gate C. Each atlas is the packed 1024x1024 image; per-slot
rects live in `*.meta.json` under `slots[NAME].pasted_rect`.

V7CompletionDataset expects `dict[skin_id -> Path]` where each Path is a
directory containing the 11 trainable BMPs at their canonical
TRAINABLE_EXPORT_SPECS dimensions.

**Open decision 1**: how to turn `atlas + meta_json` into per-skin BMP dirs.

Two options:

```text
A. New script scripts/21_unpack_atlases_to_skin_dirs.py
   - For each 16-skin atlas, crop atlas_rect from the image.
   - Resize/pad to the canonical TRAINABLE_EXPORT_SPECS (h, w).
   - Write 16 directories under data_v7_16skin_completion/{skin_id}/{file}.bmp.
   - Produces a clean, reusable on-disk dataset; matches V7CompletionDataset
     contract verbatim.

B. Extend V7CompletionDataset with an alternate constructor that loads
   targets from atlas + meta_json directly.
   - No on-disk artifacts; uses the existing data_v4_16skin/.
   - Adds complexity to the dataset class and a second code path to test.

A is cleaner. Recommendation: option A.
```

**Open decision 2**: any skin in the 16-skin set with non-canonical exported
dimensions (e.g. atlas content smaller than TRAINABLE_EXPORT_SPECS) must be
padded or rejected. The audit script needs to surface this and decide a
policy (probably: pad with the source skin's "background" color, but this
needs a defensible per-file rule).

## Training Recipe (fixed)

```bash
.venv/bin/python train_v7_completer.py \
  --state-families configs/state_families_classic.yaml \
  --skin-sources <16 entries as skin_id=path,…> \
  --steps <see step budget below> \
  --batch 1 \
  --lr 1e-3 --weight-decay 1e-4 \
  --base-channels 48 --file-embedding-dim 32 \
  --sampling-mode weighted \
  --file-sampling-weights configs/v7_file_weights_continuation.yaml \
  --sobel-weight 0.25 \
  --checkpoint-every 1000 --snapshot-every 1000 \
  --num-workers 0 --pin-memory --amp \
  --seed 0 \
  --out runs/v7_completer_gateB_16skin \
  --device cuda
```

Identical to the recipe that produced Gate A pass at step 55k. Sampling is
file-weighted, not skin-weighted: each step picks a file by weight then
samples one item from that file's 16-element group (one item per skin).

**Open decision 3**: start from scratch or warm-start from the Gate A
checkpoint?

```text
Warm-start risk: the Gate A model has memorized default_skin's exact pixels
in the file_embedding + decoder. Warm-starting biases the early training
toward "predict default_skin everywhere" until the multi-skin gradient
overwhelms that prior. May converge faster, may converge to a worse local
minimum.

Scratch advantage: cleaner Phase 0 measurement. The whole point of Gate B is
"does the recipe generalize?", not "does Gate A help bootstrap?"

User's call: scratch. Confirmed.
```

## Step Budget

Gate A passed at 55k with one skin. For 16 skins, weighted sampling still
picks files by weight; each file's group now has 16 items instead of 1, so
within-file diversity grows but per-file gradient steps stay the same per
unit time. The model has more to learn (16 distinct pixel sets per file).

```text
Initial estimate: 80-100k steps.
  Reason: Gate A converged at 55k on one skin; the model now has 16x more
  pixel content to memorize, but most learning is shared across skins
  (file structure, state-family patterns). Expect ~1.5-2x the budget, not 16x.

Decision rule for stopping: same as Gate A.
  - Eval every 5k steps.
  - First snapshot that meets aggregate sup_mae <0.015 AND hit5 >0.90 is the
    candidate accept.
  - Continue 5-10k more to confirm trajectory is not lucky.
  - If hit5 regresses for 2 consecutive evals, stop and keep best.
```

## Eval

Use the existing `scripts/19_eval_v7_completer.py`:

```bash
.venv/bin/python scripts/19_eval_v7_completer.py \
  --state-families configs/state_families_classic.yaml \
  --skin-sources <same 16 entries> \
  --checkpoint <snapshot> \
  --batch 1 --mask-samples 4 --seed 0 \
  --out-json <eval_step?????.json>
```

The eval iterates all (skin, file) items (16 * 11 = 176 items per pass) with
the default V7 mask mix. Per-file aggregates are computed across skins; a
per-skin breakdown is informational but not required for Gate B acceptance.

For Gate B we may also want to add a per-skin breakdown to the eval output.
Decision: add only if it materially helps the diagnosis when Gate B is hard.

## Reporting Required After Gate B

```text
1. Acceptance verdict against the initial thresholds.
2. Aggregate trajectory: sup_mae, hit5, sobel_mae at each eval step.
3. Per-file table at the accepted checkpoint.
4. Per-skin table at the accepted checkpoint (which skins struggle).
5. Comparison to Gate A's per-file numbers.
6. Best checkpoint path.
7. Wall time and step rate.
8. Notes on any oscillation / regression.
```

## Watch Items

Carry these forward from Gate A:

```text
- AdamW oscillation after sobel introduction. If Gate B passes around step
  K, do not blindly train past K+10k; check the eval trajectory carefully.
  Saw VOLUME hit5 0.97 -> 0.82 between Gate A 55k and 60k.

- Per-file hit5 floor: PLAYPAUS, MONOSTER, EQMAIN sat at hit5 in
  [0.91, 0.95] at Gate A's accepted checkpoint. They passed the 0.90
  floor but not the 0.95 ideal. At 16 skins, the same files may sit
  closer to the floor; may need a file-weighted hit5 floor check.

- State-family masks on per-skin variation: EQMAIN's 28 slider frames
  vary in highlight color across skins. State-family masks reveal one
  frame per draw, but across skins the same "frame 5" has different
  colors. The model needs to learn (skin_id-dependent) frame appearance
  via file_embedding alone or via cross-skin context. Skin-aware
  conditioning is currently absent.

  If Gate B fails specifically on EQMAIN per-skin variance, the natural
  next step is to add a skin embedding alongside the file embedding.
```

## Out of Scope

```text
- No observer/copy path. That is Phase 1 (Gate C).
- No retrieval eval against completer outputs.
- No new mask modes; the V7 default mix stays.
- No attention or deeper U-Net; capacity stays c48.
- No multi-resolution loss; sobel-weight stays 0.25.
```

## Pre-Launch Checklist

Before launching Gate B training:

```text
[ ] Decide on the BMP extraction script (option A above).
[ ] Run the extraction; sanity-check at least 2 skins visually.
[ ] Verify V7CompletionDataset loads the 16-skin set without error.
[ ] Re-run V7 tests against a sampled 2-skin subset.
[ ] Confirm step budget (80k / 100k) and snapshot/eval cadence.
[ ] Decide whether to add per-skin eval breakdown now or after first failure.
```

All architecture and loss decisions are locked; only data preparation and
step budget remain.
