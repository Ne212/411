#!/usr/bin/env bash
# Stage 4 fine-tuning runs (sequential on the GPU). Logs: work/survey/finetune/runs/<name>/train.log
# MACE: naive full fine-tuning of MACE-MP-0 medium (no multihead replay), foundation E0s kept (energies are
# pre-shifted per element in prepare_ft_data.py), same lr / epochs / batch / weights as CHGNet.
# --foundation_model_elements True keeps all 89 elements of the foundation model (default: only the training
# elements Au, Hg are kept, and the model then cannot evaluate Ag, Cu, ... -> no forgetting test possible).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); V=$HERE/../../../venvs
FOUND=$HOME/.cache/mace/20231203mace128L1_epoch199model
for variant in ${*:-noatom atom}; do
  name=MACE-MP-0-ft-$variant; out=$HERE/runs/$name; mkdir -p $out
  ( time $V/mace/bin/mace_run_train --name $name --work_dir $out --seed 0 --device cuda --default_dtype float64 \
      --foundation_model $FOUND --foundation_model_elements True --multiheads_finetuning False --E0s foundation \
      --train_file $HERE/data/MACE-MP-0-medium/train_$variant.xyz --valid_file $HERE/data/MACE-MP-0-medium/valid.xyz \
      --test_file $HERE/data/MACE-MP-0-medium/test.xyz \
      --energy_key REF_energy --forces_key REF_forces --loss weighted --energy_weight 1.0 --forces_weight 10.0 \
      --lr 1e-3 --batch_size 4 --valid_batch_size 4 --max_num_epochs 200 --ema --ema_decay 0.99 \
      --eval_interval 1 --patience 1000 ) > $out/train.log 2>&1
  echo "$name rc=$? $(date +%T)"
  [ "${SKIP_CHGNET:-0}" = 1 ] && continue      # used to retrain MACE only
  name=CHGNet-ft-$variant; mkdir -p $HERE/runs/$name
  ( time $V/chgnet/bin/python $HERE/train_chgnet.py $variant ) > $HERE/runs/$name/train.log 2>&1
  echo "$name rc=$? $(date +%T)"
done
