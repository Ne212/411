#!/usr/bin/env bash
# Stage 4b: MACE-MP-0 medium multihead fine-tuning WITH REPLAY of foundation (MPtrj) data.
# Same training data as MACE-MP-0-ft-atom (84 frames + free Hg atom + 'far'), same batch/weights/seed/E0s/elements.
# Differences (time budget <= 1 h): replay of NUM_PT random MPtrj structures (mace-torch downloads
# mp_traj_combined.xyz, 595 MB, from github.com/ACEsuit/mace-foundations), and EPOCHS epochs instead of 200.
# --force_mh_ft_lr True makes mace use exactly $LR.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); V=$HERE/../../../venvs
FOUND=$HOME/.cache/mace/20231203mace128L1_epoch199model
NUM_PT=${NUM_PT:-2000}; EPOCHS=${EPOCHS:-15}; LR=${LR:-1e-4}
# lr: 1e-4 = mace-torch's own multihead default. A first try with lr 1e-3 (forced, as in the other runs) was stopped
# after 2 epochs: the replay-head validation error rose (E 89 -> 203 meV/atom, F 117 -> 302 meV/A), i.e. lr 1e-3
# destroys foundation knowledge even with replay; and 30 epochs would have exceeded the 1 h budget (~3 min/epoch).
name=MACE-MP-0-ft-atom-replay; out=$HERE/runs/$name; mkdir -p $out
( time $V/mace/bin/mace_run_train --name $name --work_dir $out --seed 0 --device cuda --default_dtype float64 \
    --foundation_model $FOUND --foundation_model_elements True --multiheads_finetuning True \
    --pt_train_file mp --num_samples_pt $NUM_PT --subselect_pt random --weight_pt_head 1.0 \
    --force_mh_ft_lr True --E0s foundation \
    --train_file $HERE/data/MACE-MP-0-medium/train_atom.xyz --valid_file $HERE/data/MACE-MP-0-medium/valid.xyz \
    --test_file $HERE/data/MACE-MP-0-medium/test.xyz \
    --energy_key REF_energy --forces_key REF_forces --loss weighted --energy_weight 1.0 --forces_weight 10.0 \
    --lr $LR --batch_size 4 --valid_batch_size 4 --max_num_epochs $EPOCHS --ema --ema_decay 0.99 \
    --eval_interval 1 --patience 1000 ) > $out/train.log 2>&1
echo "$name rc=$? $(date +%T) NUM_PT=$NUM_PT EPOCHS=$EPOCHS LR=$LR"
