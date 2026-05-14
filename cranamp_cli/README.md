# Cranamp CLI Wrapper

The atlas pack/export tooling in this repo is usable without a Cranamp CLI.
Renderer-driven dataset generation still needs a `cranamp-cli` binary or wrapper
that exposes the roadmap commands:

- `dump-classic-spec`
- `render-random`
- `render-with-params`

Until that exists, point tools and notes at the local Cranamp checkout with:

```bash
export CRANAMP_REPO=/home/s/develop/projects/cranamp
```

The configs currently use the local Cranamp source constants and the supplied
default skin dimensions. In particular, `TITLEBAR.bmp` is `344x87`; the atlas
slot is wider than the original roadmap placeholder so export can round-trip the
actual asset.
