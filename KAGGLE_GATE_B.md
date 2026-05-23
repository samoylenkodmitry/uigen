# Kaggle launch sequence — V7 Gate B

Step-by-step recipe for running the 14-skin completion training on Kaggle
Notebooks. Do not launch Gate B until every step in this file has passed.

## 0. Local artefact

The dataset archive used in steps 1–2 is built from this repo with:

```sh
make package-kaggle-data   # or, equivalently:
tar -czf dist/uigen-data-v7-16skin.tar.gz \
    --exclude='*.pyc' --exclude='__pycache__' \
    data_v7_16skin_completion
sha256sum dist/uigen-data-v7-16skin.tar.gz
```

Current archive: `dist/uigen-data-v7-16skin.tar.gz`
SHA256 in `dist/uigen-data-v7-16skin.tar.gz.sha256`.

Contents — verify before upload:

- 14 skin directories, each holding 11 canonical BMPs + `_meta.json`
- top-level `_manifest.json` (16 skins listed; 14 accepted, 2 skipped with
  reason: darkside / PLAYPAUS 1x1, simblyblayit / POSBAR 3x1)
- no run outputs, no checkpoints, no .venv

## 1. Create the Kaggle dataset

1. Kaggle → *Datasets* → *New dataset*.
2. Title: `uigen-data`.
3. Drop `dist/uigen-data-v7-16skin.tar.gz` and the `.sha256` file.
4. Kaggle auto-extracts tarballs; after publish the layout is:

   ```
   /kaggle/input/uigen-data/data_v7_16skin_completion/
       _manifest.json
       <14 skin dirs>/
           *.bmp
           _meta.json
   ```

This matches `configs/runtime/kaggle.yaml` (`data_dir: /kaggle/input/uigen-data`).

## 2. Notebook bootstrap

In a fresh notebook with the dataset attached:

```sh
%cd /kaggle/working
git clone https://github.com/samoylenkodmitry/uigen.git
%cd uigen
git rev-parse HEAD   # must be >= 36c0182 (per-skin accounting fix)

pip install -r requirements.txt   # if a requirements file exists; else:
pip install torch safetensors pillow numpy pyyaml
```

## 3. Sanity probes (must all pass before launch)

```sh
# (a) Environment
python scripts/print_env.py

# (b) Resolved command
make kaggle-dry EXPERIMENT=experiments/v7_completer_gateB.yaml
# expected: --skin-sources lists all 14 accepted skin dirs under
# /kaggle/input/uigen-data/data_v7_16skin_completion/
# expected: --out /kaggle/working/runs/v7_gateB_16skin

# (c) Short benchmark
make bench-kaggle EXPERIMENT=experiments/v7_completer_gateB.yaml
# expected: sec/step printed, peak VRAM under ~2 GiB for c48.
```

If any of (a)–(c) fail or the resolved skin list is short, stop and
investigate. Do **not** continue to step 4.

## 4. Gate B launch (only after step 3 is clean)

```sh
make kaggle EXPERIMENT=experiments/v7_completer_gateB.yaml
```

Or equivalently:

```sh
UIGEN_RUNTIME=configs/runtime/kaggle.yaml \
python scripts/run_experiment.py \
    --experiment experiments/v7_completer_gateB.yaml
```

The experiment YAML already pins: 80 000 steps, base_channels=48, weighted
sampler, sobel_weight 0.25, save/eval every 5 000 steps.

## 5. What to watch

Every 5 000 steps the trainer writes a checkpoint and the eval script
reports four levels of metrics. Watch all four:

- aggregate `supported_mae` — should decrease monotonically.
- aggregate `hit5` — should rise into the >0.95 band.
- `per_file` — slider strips (VOLUME, BALANCE) are the slowest; if
  hit5 stays under 0.80 on those past 40k, sobel weight may need a bump.
- `per_skin` — flag any skin with `supported_mae` more than 2x the
  median; the new accounting (commit 36c0182) makes this signal
  trustworthy even when same-file batches mix skins.

Gate B acceptance (mirror of Gate A but across the 14-skin set):

- aggregate `supported_mae` < 0.005
- aggregate `hit5` > 0.95
- no per-file `supported_mae` > 0.010
- no per-skin `supported_mae` > 0.010
