# Project conventions for AI agents

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

## Kaggle CLI: no live stop, no live stdout

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
