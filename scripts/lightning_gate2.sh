#!/usr/bin/env bash
# V10 Gate 2 on a Lightning.ai Studio (or any GPU box). Self-contained: extracts
# the shipped 14-skin source, generates the multi-skin dataset, then runs the
# per-expert L1->adversarial sweep. No secrets needed (skins ride in the repo).
#
# One-paste in a fresh GPU studio terminal:
#   git clone https://github.com/samoylenkodmitry/uigen.git && cd uigen && \
#     nohup bash scripts/lightning_gate2.sh > gate2.log 2>&1 & tail -f gate2.log
#
# Env overrides:
#   SCALE=smoke EXPERTS=MAIN bash scripts/lightning_gate2.sh   # ~5min sanity run
#   SCALE=gate2 EXPERTS=ALL  bash scripts/lightning_gate2.sh   # full Gate 2 (default)
set -euo pipefail
cd "$(dirname "$0")/.."

SCALE="${SCALE:-gate2}"
EXPERTS="${EXPERTS:-ALL}"
DATA="data_v10_${SCALE}_l"
RUN="runs/gate2"
PY="${PY:-python}"

echo "### V10 Gate 2 bootstrap | scale=$SCALE experts=$EXPERTS $(date -u +%FT%TZ)"
$PY - <<'PYCHK'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PYCHK

echo "### 1/4 deps (torch assumed preinstalled on the GPU image)"
$PY -m pip install -q safetensors pyyaml pillow numpy || true

echo "### 2/4 extract 14-skin source"
[ -d data_v7_16skin_completion ] || tar xzf assets/v10/skins14.tar.gz
ls -d data_v7_16skin_completion/*/ | wc -l | xargs echo "skin dirs:"

echo "### 3/4 generate $SCALE dataset across 14 skins -> $DATA"
SKINS=(
  "aguileramp_oldschool_2e1e7540:aguileramp_oldschool"
  "a_halo_so_bright_it_bleeds_3ee84993:a_halo_so_bright"
  "blair_razor_project_e7dd3210:blair_razor"
  "cyborg_5569c5cf:cyborg"
  "dragonzv30amp_85acc35c:dragonzv30amp"
  "engraved4_platinum_5638acf5:engraved4_platinum"
  "goodgawd_bba84deb:goodgawd"
  "infected_fx_gray_no_transparency_9f3bd211:infected_fx_gray"
  "minimalistic_black_145917e6:minimalistic_black"
  "rancid_amp_5_42a78437:rancid_amp_5"
  "ruki2_by_michi_caa5bfe3:ruki2_by_michi"
  "the_four_horsemen_523e6bdf:the_four_horsemen"
  "tvxq_winamp_skins_by_roseweedy_c379f7bd:tvxq_roseweedy"
  "zelda_amp_gold_3cc38af4:zelda_amp_gold"
)
if [ ! -f "$DATA/csv/train_MAIN.csv" ]; then
  rm -rf "$DATA"; i=0
  for entry in "${SKINS[@]}"; do
    dir="${entry%%:*}"; sid="${entry##*:}"
    args=(--skin "data_v7_16skin_completion/$dir" --skin-id "$sid" --scale "$SCALE" --out "$DATA" --seed "$i" --progress-every 200)
    [ "$i" -gt 0 ] && args+=(--append)
    echo "  gen $((i+1))/14: $sid"
    $PY scripts/make_v10_bmp_expert_dataset.py "${args[@]}"
    i=$((i+1))
  done
else
  echo "  (dataset already present, skipping gen)"
fi
rows=$(( $(wc -l < "$DATA/csv/train_MAIN.csv") - 1 ))
echo "  train_MAIN.csv rows=$rows across 14 skins"

echo "### 4/4 sweep ($EXPERTS) -> $RUN"
$PY scripts/v10_gate_sweep.py --data "$DATA" --out "$RUN" --device cuda --experts "$EXPERTS"

echo "### DONE. checkpoints: $RUN/ckpts/<EXPERT>/last.safetensors ; verdicts: $RUN/summary.json"
