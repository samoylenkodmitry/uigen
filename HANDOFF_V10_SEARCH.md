# V10 architecture search — hard facts & constraints (2026-05-30)

Persistent ground truth for the V10 Gate-2+ architecture search. Read this first.

## The product vision (FIRM — do not violate)
- One ML model **per component** (per output BMP: MAIN, TITLEBAR, CBUTTONS,
  MONOSTER, PLAYPAUS, POSBAR, VOLUME, BALANCE, SHUFREP, EQMAIN, PLEDIT).
- Input = the **WHOLE** arbitrary skin mockup image (anything an image generator
  produces — e.g. a cat-themed "KITTENAMP"). **NEVER crop / slice / region-select
  the input.** The whole image goes to every component model. (Manual per-window
  crop was tried 2026-05-30 and rejected by the user — twice. Do not reintroduce.)
- Each component model must **imagine its full sprite atlas including all hidden
  states** (every EQ thumb position, every button state) in the input's style —
  image-gen-like imagination, not copying. The adversarial fine-tune already
  proved this for EQMAIN's hidden slider rows.
- Generalization needs diversity on **two axes**: (1) many **skins/styles**,
  (2) many **transformations per skin** (state + geometry/scale jitter) — so the
  model learns to *fit what's fittable and imagine what's hidden* at any layout.

## Resources
- **Local RTX 2070 8GB: FREE + UNLIMITED.** Compute-bound on the full 1728x960
  input; batch ≤2 at base config. Use it as the workhorse for cheap probes and
  back-to-back experiments — it costs nothing.
- **Lightning.ai: ~70 GPU-HOURS FREE (precious).** Studio `scratch-studio-devbox`,
  teamspace `gpu-model-development-project`, user `trdmitriisamoilenko`. Driven
  via `lightning_sdk` in `.venv` (already authed). Gen datasets on a CPU machine
  (free of GPU hrs), train on GPU. A100 NOT available on the cluster; **L40S** is
  (48GB, ~2.8x local-equivalent throughput). `s.run()` raises on any nonzero exit
  → append `|| true` to greps. STOP the studio between runs to halt billing.
- **Skins: `skins_raw/` has 7,957 `.wsz` source skins** — the diversity goldmine.
  A `.wsz` is a ZIP of BMPs (case-insensitive names). Extract via
  `scripts/extract_wsz_skins.py`. `data_v7_16skin_completion/` = 14 already
  materialized; `assets/v10/skins14.tar.gz` ships those for cloud bootstrap.

## Budget rules (HARD)
- **≤ 1 HOUR per training run.** Never exceed (project rule, reaffirmed).
- The ~70 Lightning hrs are for **ARCHITECTURE SEARCH** (find the final
  architecture via cheap 1hr probes), **NOT** training to convergence. Selection
  metric = learning EFFICIENCY in 1hr (eval-mae trajectory). Prefer the FREE
  local 2070 for probes; reserve Lightning hrs for runs that need the speed/scale.
- After the architecture is locked: the user buys GPU and does ONE paid
  full-train (full data, full steps), spend **≤ $100**. Deliverable of the free
  phase = locked architecture + recipe + a costed full-train plan.

## What is already proven
- Gate 1 (one-skin overfit, minimalistic_black): ALL 11 experts PASS (mae<0.01,
  hit5>0.90). 8 via L1; EQMAIN/BALANCE/TITLEBAR via adversarial fine-tune.
- **Adversarial fine-tune is the hidden-state "imagination" engine** (L1 alone
  seeks the blurry mean → smears 1px detail/flat color; the discriminator forces
  crisp plausible detail). Recipe: pretrain L1, then `--init-from` + `--adversarial
  --adv-weight 0.02 --fm-weight 1.0 --d-lr 2e-4 --lr 1e-4`, FP32.
- Gate 2 (14 skins, multi-skin generalization): the BASE architecture DOES
  generalize (MAIN on L40S full data: mae 0.10→~0.01, 8/14 skins pass, still
  descending) — it is **compute/step-limited, NOT a capacity wall**. Bottleneck =
  encoder convs on the full 1728x960 input (the KV is a fixed adaptive pool, so
  ONLY the encoder scales with input size; SM util ~99-100%).

## Gate metric
- Per-skin: a skin passes if mae<0.01 AND hit5>0.90. `eval_bmp_expert.py` reports
  `per_skin`, `gate2_pass` (= EVERY skin passes), `worst_skin`. Multi-skin runs
  use `gate2_pass`; single-skin uses `gate1_pass`.

## Key tooling
- Trainer: `train_bmp_expert.py` (`--init-from`, `--adversarial`, `--max-minutes`,
  `--early-stop`, `--amp`, capacity flags). AMP+batch1 diverges; use batch≥2 or FP32.
- Eval: `scripts/eval_bmp_expert.py` (per-skin gate).
- Sweep orchestrator: `scripts/v10_gate_sweep.py` (per-expert L1→adversarial
  triage, auto-batch from VRAM, capacity flags).
- Dataset gen: `scripts/make_v10_bmp_expert_dataset.py` (per-skin state+geometry
  sweeps, `--append` for multi-skin). Scales: smoke/gate1/gate2.
- Lightning bootstrap: `scripts/lightning_gate2.sh`.

## Search findings (live)
- **s16 (2026-05-30): MAIN trained on 16 skins MEMORIZES, does NOT generalize.**
  train mae 0.038 vs HELD-OUT (16 disjoint skins) mae **0.311**, hit5 0.03 — an 8x
  gap. Local 2070, smoke views, 55min/11k steps. Held-out datasets:
  data_v10_train{16,64}, data_v10_held16 (disjoint, from data_v10_skins256).
- **HYPOTHESIS to test (architectural):** MAIN.bmp is largely VISIBLE in the
  render (main window shows it), so recovering it for an unseen skin is mostly a
  geometric un-transform of the visible region + imagine-occluded — this SHOULD
  generalize. The fixed, lossy KV pool (DEFAULT_KV_POOL 48x28..18x10, ~2.4k
  tokens) may destroy the spatial detail needed to RECONSTRUCT an unseen detailed
  BMP, so the decoder memorizes seen skins instead. If s64 also fails held-out,
  test architectures that preserve spatial detail (larger/higher-res KV pool, or
  encoder→decoder skip/U-Net path) BEFORE concluding "need more skins". This is a
  real architecture-search axis and is fully consistent with whole-input/no-crop.
- **s64 (2026-05-30): 64 skins -> held-out mae 0.261 (hit5 0.007), train 0.109.**
  vs s16 held 0.311/train 0.038. Diversity helps held-out monotonically
  (0.311->0.261) BUT both runs are badly UNDER-TRAINED (55min/8-11k steps on 2070;
  one skin alone needed ~7k; s64 hasn't fit train). So the generalization signal
  is real-but-weak and CONFOUNDED by throughput. Can't yet separate "arch can't
  generalize" from "not converged". The 2070 + 55min is under-powered for a clean
  multi-skin generalization read. Next: a CLEANER test (faster GPU to converge, or
  detail-preserving arch A/B) — pending user steer on resource allocation.

## Search plan (current)
1. Materialize a diverse N-skin set from `skins_raw/` (scale 14→60→… as needed).
2. Data-diversity scaling: does generalization keep improving as skins +
   transform variety grow? (the core vision question)
3. Capacity + recipe tuning within ≤1hr probes.
4. Lock architecture + recipe; write the ≤$100 paid full-train plan.
5. Gate 4 product validation: feed an arbitrary imagegen mockup (KITTENAMP) to
   the trained experts; expect a coherent per-component atlas in that style.
