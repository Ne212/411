#!/usr/bin/env bash
# Stage 4b control: MACE-MP-0 medium, naive fine-tuning WITHOUT replay, same data as MACE-MP-0-ft-atom,
# but lr 1e-4 and 15 epochs = the settings of the replay run. Separates the effect of replay from that of the lower lr.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); V=$HERE/../../../venvs
FOUND=$HOME/.cache/mace/20231203mace128L1_epoch199model
name=MACE-MP-0-ft-atom-lr1e-4; out=$HERE/runs/$name; mkdir -p $out
( time $V/mace/bin/mace_run_train --name $name --work_dir $out --seed 0 --device cuda --default_dtype float64 \
    --foundation_model $FOUND --foundation_model_elements True --multiheads_finetuning False --E0s foundation \
    --train_file $HERE/data/MACE-MP-0-medium/train_atom.xyz --valid_file $HERE/data/MACE-MP-0-medium/valid.xyz \
    --test_file $HERE/data/MACE-MP-0-medium/test.xyz \
    --energy_key REF_energy --forces_key REF_forces --loss weighted --energy_weight 1.0 --forces_weight 10.0 \
    --lr 1e-4 --batch_size 4 --valid_batch_size 4 --max_num_epochs 15 --ema --ema_decay 0.99 \
    --eval_interval 1 --patience 1000 ) > $out/train.log 2>&1
echo "$name rc=$? $(date +%T)"
