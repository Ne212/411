#!/usr/bin/env bash
# One venv per MLIP under START/venvs/<name>.  Usage: bash install_envs.sh [name ...]
# Every install is logged to results/survey_logs/install_<name>.log and frozen to freeze_<name>.txt.
#
# Torch choice: the RTX PRO 4000 Blackwell is sm_120, which needs CUDA >= 12.8 wheels.
# We use the cu130 wheels from download.pytorch.org (the driver reports CUDA 13.2).
# Default torch = 2.11.0+cu130, the version already proven on this GPU in the 411 venv.
# Only fairchem-core 2.25 requires a different one (torch~=2.13.0).
set -u
ROOT=/home/ne212/START
UV=$ROOT/venvs/_tools/bin/uv
LOG=$ROOT/results/survey_logs
TORCH_IDX=https://download.pytorch.org/whl/cu130
mkdir -p "$LOG"

mk() {  # mk <name> <python> <torch spec or "none"> <packages...>
  local name=$1 py=$2 torchspec=$3; shift 3
  local V=$ROOT/venvs/$name
  {
    echo "=== $name  $(date '+%F %T')  python=$py torch=$torchspec pkgs=$*"
    rm -rf "$V"
    # PYFLAGS: e.g. --managed-python (uv's own CPython, ships Python.h; triton needs it to build its driver)
    $UV venv -q ${PYFLAGS:-} --python "$py" "$V" || return 1
    if [ "$torchspec" != none ]; then
      $UV pip install -p "$V/bin/python" --index-url $TORCH_IDX $torchspec || return 1
    fi
    # PyPI for the MLIP itself; torch stays the cu130 build already installed
    # (the constraint file stops a resolver from swapping it for a CPU/cu12 wheel).
    "$V/bin/python" -c "import torch;print('torch=='+torch.__version__)" > "$V/constraints.txt" 2>/dev/null || : > "$V/constraints.txt"
    $UV pip install -p "$V/bin/python" -c "$V/constraints.txt" \
        --extra-index-url $TORCH_IDX --index-strategy unsafe-best-match "$@" || return 1
    # common analysis stack for the survey scripts (ase pinned to the 411 version)
    $UV pip install -p "$V/bin/python" -c "$V/constraints.txt" \
        --extra-index-url $TORCH_IDX --index-strategy unsafe-best-match $COMMON || return 1
    $UV pip freeze -p "$V/bin/python" > "$LOG/freeze_$name.txt"
    echo "=== $name OK $(date '+%F %T')"
  } > "$LOG/install_$name.log" 2>&1 && echo "$name: OK" || echo "$name: FAILED (see $LOG/install_$name.log)"
}

T211="torch==2.11.0"
COMMON="ase==3.29.0 pandas matplotlib scipy"
want=${*:-mace chgnet matgl alignn sevenn mattersim orb fairchem grace nequip deepmd upet}
for n in $want; do
  case $n in
    mace)      mk mace      3.11 "$T211" mace-torch==0.3.17 ;;
    chgnet)    mk chgnet    3.11 "$T211" chgnet==0.4.2 ;;
    matgl)     mk matgl     3.11 "$T211" matgl==4.1.4 ;;
    alignn)    mk alignn    3.11 "$T211" alignn==2026.11.1 ;;
    sevenn)    mk sevenn    3.11 "$T211" sevenn==0.13.0 ;;
    mattersim) mk mattersim 3.12 "$T211 torchvision==0.26.0 torchaudio==2.11.0" mattersim==1.2.5 ;;
    orb)       mk orb       3.12 "$T211" orb-models==0.7.0 ;;
    fairchem)  PYFLAGS=--managed-python mk fairchem 3.12 "torch==2.13.0" fairchem-core==2.25.0 ;;  # torch 2.13 native ops use triton
    grace)     mk grace     3.11 none tensorpotential==0.6.2.1 ;;
    nequip)    mk nequip    3.11 "$T211" nequip==0.19.1 nequip-allegro==0.8.3 ;;
    deepmd)    mk deepmd    3.11 "$T211" "deepmd-kit[torch]==3.2.0" ;;   # [torch] extra brings e3nn, vesin, mpich, nvalchemi ops
    upet)      mk upet      3.11 "$T211" upet==0.3.1 ;;
    *) echo "unknown env $n" ;;
  esac
done
