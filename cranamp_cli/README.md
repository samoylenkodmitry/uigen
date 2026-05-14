# Cranamp CLI Wrapper

`cranamp_cli/cranamp/` is a vendored copy of the local Cranamp checkout used as
the renderer fork for this project. The original checkout remains untouched.

`cranamp_cli/cranamp-cli` is the current executable entrypoint. It implements
the roadmap commands:

- `dump-classic-spec`
- `render-random`
- `render-with-params`

The first implementation is a deterministic classic-skin compositor built from
Cranamp's Winamp sprite constants and skin loading behavior. It renders main,
EQ, and playlist windows, writes `[80,5]` rect labels, `[32]` state vectors,
replayable `params.json`, and a visible atlas mask derived from the source slot
pixels that were blitted.

Run a smoke render:

```bash
./cranamp_cli/cranamp-cli render-random \
  --skin-dir assets/default_skin \
  --seed 1 \
  --canvas-w 768 \
  --canvas-h 1280 \
  --out-view /tmp/cranamp_view.png \
  --out-rects /tmp/cranamp_rects.f32 \
  --out-state /tmp/cranamp_state.f32 \
  --out-visible-atlas-mask /tmp/cranamp_mask.png \
  --out-params /tmp/cranamp_params.json \
  --state-balanced false
```

The renderer is intentionally isolated in the copied fork so future work can
replace the Python compositor with direct Cranamp renderer instrumentation
without changing the dataset script contract.
