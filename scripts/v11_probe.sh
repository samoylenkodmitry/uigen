#!/usr/bin/env bash
# Resumable V11 per-component probe. KILL ANYTIME -> re-run the SAME command and it
# CONTINUES from the last checkpoint (--resume + checkpoint-every 1500 + no out-dir
# wipe). --steps is set very high; each launch trains until --max-minutes (<=1hr
# rule), so repeated launches accumulate progress. Judged by cond_eval (own/shuffled/
# diversity) + held-out grid, not exact MAE alone.
#
# Env knobs (all optional):
#   BMP=CBUTTONS.bmp  TRAIN=data_v10n_train64  HELD=data_v10n_held16
#   OUT=/tmp/v11_cbuttons  MAXMIN=45
#   EXTRA="--color-aug --encoder convnext --adversarial --cond-disc --init-from <ckpt> --lr 1e-4 \
#          --adv-weight 0.03 --fm-weight 0.5 --d-lr 4e-4 --d-base 48 --d-layers 2"
set -u
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv/bin/python}"
BMP="${BMP:-CBUTTONS.bmp}"; TRAIN="${TRAIN:-data_v10n_train64}"; HELD="${HELD:-data_v10n_held16}"
OUT="${OUT:-/tmp/v11_${BMP%.bmp}}"; MAXMIN="${MAXMIN:-45}"; EXTRA="${EXTRA:-}"
echo "### V11 probe (resumable) bmp=$BMP train=$TRAIN out=$OUT maxmin=$MAXMIN extra=[$EXTRA]"
$VENV train_bmp_expert.py --data "$TRAIN" --bmp "$BMP" --out "$OUT" --resume \
  --steps 1000000 --batch 6 --base 48 --attn-dim 256 --dec-ch 128 --heads 4 --attn-layers 2 \
  --query-div 4 --decoder progressive --checkpoint-every 1500 \
  --max-minutes "$MAXMIN" --eval-every 3000 --eval-max-items 192 --progress-every 1000 \
  --num-workers 3 --device cuda $EXTRA
$VENV scripts/eval_bmp_expert.py --data "$HELD" --bmp "$BMP" \
  --checkpoint "$OUT/best.safetensors" --out "$OUT/eval_held" --batch 16 --grid-samples 16 --device cuda >/dev/null 2>&1 || true
echo -n ">>> "; $VENV scripts/cond_eval.py --data "$HELD" --bmp "$BMP" --checkpoint "$OUT/best.safetensors" --device cuda || true
echo "V11_PROBE_DONE ($OUT)"
