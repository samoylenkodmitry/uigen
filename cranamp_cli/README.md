# Cranamp CLI Wrapper

`cranamp_cli/cranamp/` is a vendored copy of the local Cranamp checkout used as
the renderer fork for this project. The original checkout remains untouched.

`cranamp_cli/cranamp-cli` is the current executable entrypoint. It implements
the roadmap commands:

- `dump-classic-spec`
- `render-random`

The first implementation is a deterministic classic-skin compositor built from
Cranamp's Winamp sprite constants and skin loading behavior. It renders main,
EQ, and playlist windows. For the V3.4 training pipeline, the only persisted
training input is the rendered PNG; random render parameters are intentionally
not saved.

Run a smoke render:

```bash
./cranamp_cli/cranamp-cli render-random \
  --skin-dir assets/default_skin \
  --seed 1 \
  --canvas-w 941 \
  --canvas-h 1672 \
  --out-view /tmp/cranamp_view.png
```

The renderer is intentionally isolated in the copied fork so future work can
replace the Python compositor with direct Cranamp renderer instrumentation
without changing the dataset script contract.
