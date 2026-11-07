#!/usr/bin/env bash
# Stage 4 evaluation of the fine-tuned models, with the SAME scripts and settings as Stages 2-3:
#   1. frame errors on the 411 QE splits     (eval_frames.py)
#   2. Hg/Au(111) 411 protocol               (hg_au_protocol.py: E_ads 4 sites, free-atom error, Au a0/B0)
#   3. Steckel 2008 test set, 44 systems     (published_benchmark.py: transfer to Ag, Cu, Ni, Pd, Pt = forgetting check)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); V=$HERE/../../../venvs
KEYS="MACE-MP-0-ft-noatom MACE-MP-0-ft-atom CHGNet-ft-noatom CHGNet-ft-atom"
for k in $KEYS; do
  case $k in MACE*) env=mace; base=MACE-MP-0-medium ;; CHGNet*) env=chgnet; base=CHGNet-0.3.0 ;; esac
  $V/$env/bin/python $HERE/eval_frames.py $k $base > $HERE/runs/$k/eval_frames.log 2>&1; echo "$k frames rc=$?"
done
python3 $HERE/../run_all_hg_au.py --task hg_au --models $KEYS
python3 $HERE/../run_all_hg_au.py --task published --models $KEYS
