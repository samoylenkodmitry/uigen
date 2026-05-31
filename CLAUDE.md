# Project conventions for AI agents

## V11 (current): architecture SEARCH COMPLETE — read HANDOFF_V11_ARCH.md first

One ML model per skin component (per BMP); input = the WHOLE arbitrary skin mockup
(never cropped); each model imagines its full sprite atlas incl. hidden states in the
input's style. **The V11 search is DONE and the recipe is LOCKED** — full design,
every probe, and the finalized PAID FULL-TRAIN RUNBOOK live in **HANDOFF_V11_ARCH.md**
(read it first). LOCKED recipe: native-res input + paired color-aug + GLOBAL cond-disc
fine-tune + MAX unique skins (spatial cond-D refuted/off). Next step = the ≤$100 paid
full-train (user buys GPU; runbook in the handoff). V10 history: HANDOFF_V10_SEARCH.md.
Non-negotiables: NEVER crop the input; native-res (not 960×1728); cond-disc as FT not
step 0; the local RTX 2070 is free+unlimited; `skins_raw/` holds 7,787 source skins
(the diversity asset); paid full-train ≤$100 (est. ~$27–39 on RunPod L40S).

## Long-running scripts MUST emit progress to stdout

Any script that runs for more than ~30 seconds — training loops, eval
sweeps, dataset materialization, multi-step benchmarks, batch downloads
— **must** print live progress to stdout. Writing only to a JSON/JSONL
log file is not enough.

Why: when these scripts run remotely (Kaggle script kernels, Colab,
Cron jobs, CI) the only signal we have during execution is stdout.
File-based logs are only delivered after the run finishes (or is
cancelled). A trainer with no stdout output is opaque — there is no
way to tell whether it is still progressing, stalled, or stuck in I/O,
and ETAs become guesses.

Required for every long-running script:

1. One line at start summarizing what's about to run (target steps,
   batch, key hyperparameters). Print before the first heavy op.
2. A periodic progress line every N units of work (steps, files,
   samples) showing at minimum:
     - current/total + percent
     - the metric being optimized (loss, mae, etc.), preferably a
       running mean over the last N units (not a single noisy point)
     - sec per unit, measured from the wall clock since the last print
     - elapsed time + ETA derived from sec/unit and remaining units
3. `flush=True` on every progress line — stdout is line-buffered when
   piped, which is the case in every remote runner.
4. One line at end summarizing final state (steps actually run, best
   metric, reason for stopping).

Cadence rule: pick N so the progress line lands every 30–60 s in the
common runtime. Faster than 30 s is noise; slower than 60 s leaves the
log silent long enough to look hung.

Expose the cadence as a CLI flag (e.g. `--progress-every N`) so callers
can tune it for short benchmarks (N=20) versus long training (N=200).
`0` may disable it; do not silently default to 0.

When in doubt: print more, not less. A handful of extra log lines per
minute costs nothing; one un-observable 4-hour run wastes a quota.

## Kaggle CLI: no live stop, no live stdout (workaround: scripts/kaggle_live_log.py)

`kaggle kernels` exposes `push/status/output/logs/delete` but no
`cancel`/`stop`. To halt a running script kernel you must use the
Kaggle web UI. `kaggle kernels logs` returns empty while the run is
live — the log is only persisted with the rest of `/kaggle/working`
after the kernel exits (or is cancelled via UI). On UI cancel the
output IS persisted, so checkpoints saved during the run survive and
can be downloaded with `kaggle kernels output`.

Plan for it: save checkpoints frequently enough that a forced cancel
loses at most a tolerable number of steps. The V7 trainer defaults
(`--checkpoint-every 5000`, `--snapshot-every 10000`) bound this at
≤5000 steps.

For live progress while a kernel is still running, use
`scripts/kaggle_live_log.py owner/slug`. It snapshots the user's
`~/.config/thorium` profile, launches a headless `thorium-browser`
against the snapshot with CDP enabled on `127.0.0.1:9222`, navigates
to the kernel page (the session cookie comes from the snapshotted
profile — the portal app-id matches the original binary so cookies
decrypt), pulls every visible text node, filters for trainer progress
lines, and dedupes by step number. Reuses an already-running headless
instance if present. Requires `--progress-every N` on the trainer to
actually be emitting step lines.
