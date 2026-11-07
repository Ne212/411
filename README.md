# Hg adsorption on Au(111): MLIP vs DFT benchmark (JINR INTEREST project 411)

Neşet Koçkar (Department of Physics, Ankara University). JINR INTEREST programme, Wave 15 (Oct–Dec 2026).
Supervisors: Dr Miroslav Iliaš, Dr Dipayan Sen (BLTP, JINR).

Adsorption energy of a single Hg atom (the lighter homologue of copernicium) on Au(111), computed with two universal machine-learning interatomic potentials (CHGNet, MACE-MP-0) and plane-wave DFT (Quantum ESPRESSO, PBE and PBE-D3(BJ)) using one consistent protocol. Report: [`report/main.pdf`](report/main.pdf).

| E_ads (eV), isolated-atom reference | ontop | bridge | fcc | hcp |
|---|---|---|---|---|
| CHGNet 0.3.0 | −0.734 | −0.907 | −0.943 | −0.939 |
| MACE-MP-0 medium | −0.574 | −0.748 | −0.774 | −0.768 |
| DFT PBE | −0.266 | −0.315 | −0.333 | −0.333 |
| DFT PBE-D3(BJ) | −0.579 | −0.660 | −0.686 | −0.685 |
| Experiment (thermochromatography) | | | −1.02 ± 0.03 (98 ± 3 kJ/mol) | |

Main finding: relative to PBE, the MLIPs' free Hg atom energy is too high by +0.45 eV (CHGNet) and +0.22 eV (MACE). This makes their adsorption energies too negative by the same amount. A "far-away" reference does not detect the error, because the models' cutoff is 6 Å. Full tables: `results/comparison_table.csv`.

**Not included in this repository:** `upstream/` (use https://github.com/miroi/open-collection, commit 31fe4df), `venv/` and `tools/` (rebuild from `requirements.txt`), pseudopotential files (see `work/dft/pseudo/README.md`), and the intermediate `.traj` trajectories (final structures are in `results/structures/mlip/*.xyz`). Paths in the scripts and QE inputs are those of the original workstation.

License: MIT (see `LICENSE`); the scripts in `work/upstream_runs/` are © Miro Iliaš, MIT.

**MLIP survey (November 2026, follow-up):** 31 ASE-compatible MLIP models/tasks from 12 packages tested on Hg/Au(111),
against published VASP-PBE data (Steckel 2008, Hg on six metals) and with a fine-tuning feasibility test (MACE, CHGNet).
Report: [`report/mlip_survey.md`](report/mlip_survey.md); talk outline: [`report/talk_outline.md`](report/talk_outline.md);
how to reproduce: section "MLIP survey" below.


## Project layout (original workstation)

```
upstream/   sparse clone of miroi/open-collection (adsorptions_with_ase only), commit 31fe4df, NOT modified
venv/       Python venv (inherits the local CUDA torch + MACE env, adds chgnet)
work/       all scripts written for this study
results/    CSV/JSON outputs, structures, figures
report/     LaTeX report (main.tex + figures/), Overleaf-ready
```

## 1. Environment (as found on this machine, 2026-10-29)

| Item | Found |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24 GB, driver 595.99.02 |
| CPU | Xeon w5-2445, 10 cores / 20 threads |
| Python venv | `venv/` = `python -m venv --system-site-packages` on `~/Çalışmalar/Programlar/MACE/env` (Python 3.11.17, torch 2.11.0+cu130, CUDA available, mace-torch 0.3.16, ase 3.29.0, pandas 3.0.6, matplotlib 3.11.2) + `pip install chgnet` (chgnet 0.4.2, weights 0.3.0) |
| `pw.x` (launcher in `~/Çalışmalar/Çalıştır`) | **QE 7.0** (conda), MPI + OpenMP, Open MPI 4.1.6. The local readme says 7.4, but the binary reports 7.0 |
| `pw-mkl.x` | QE 7.6, CPU, MPI, MKL |
| `pw-gpu.x` | QE 7.6, CUDA (NVHPC 26.9, cuFFT/cuBLAS linked), MPI |
| MPI | the launchers' own `mpirun`; parallel runs via `NP=<n> pw.x ...` |
| Pseudopotentials (local) | SSSP 1.3 PBE *precision*: Au `Au_ONCV_PBE-1.0.oncvpsp.upf` (SG15 ONCV, NC, scalar-relativistic, 50/200 Ry), Hg `Hg.us.z_12.uspp.gbrv.v1.upf` (GBRV USPP, scalar-relativistic, 65/390 Ry) |
| git | not on PATH; `~/Çalışmalar/Programlar/_derleme/araclar/bin/git` was used for the clone |

Recreate the venv:
```bash
~/Çalışmalar/Programlar/MACE/env/bin/python -m venv --system-site-packages venv
./venv/bin/pip install chgnet
```

## 2. Reproducing the upstream MLIP numbers

Copies of the upstream scripts, byte-identical (md5 checked), are run from `work/upstream_runs/<name>/`, so the upstream tree is not written to:
```bash
cd work/upstream_runs/chgnet && CUDA_VISIBLE_DEVICES="" ../../../venv/bin/python Hg_on_Au-slab_chgnet.py > log.txt 2>&1
cd work/upstream_runs/mace   && ../../../venv/bin/python Hg_on_Au-slab_ML_03.py > log.txt 2>&1
cd work/upstream_runs/lj     && ../../../venv/bin/python Hg_on_Au-slab_LJ.py > log.txt 2>&1
```
| Script | Upstream log | This machine |
|---|---|---|
| CHGNet (3x3x4, bottom 2 fixed, fmax 0.05, a = 4.08) | fcc −0.8986, ontop −0.7069 | fcc −0.8986, ontop −0.7069 (CPU and GPU identical) |
| MACE (4x4x4, **bottom 1** fixed, fmax 0.005) | ontop −0.5678, bridge −0.7084, fcc −0.7288, hcp −0.7249 | identical |
| LJ (ASE defaults ε = 1 eV, σ = 1 Å) | −0.0108 | −0.0108 |

## 3. Consistent MLIP benchmark

```bash
cd work
../venv/bin/python mlip_benchmark.py       # ~7 min on the GPU -> results/mlip_*.csv, results/structures/mlip/
../venv/bin/python mlip_sensitivity.py     # ~5 min -> results/mlip_sensitivity.csv
../venv/bin/python plot_results.py         # figures -> report/figures, results/figures
```
Protocol (identical for every method, all settings logged in `results/mlip_settings.json` and `results/mlip_log.txt`):
* Au fcc a0 from that method (bounded Brent on E(a), then Birch–Murnaghan on ±3 %, 11 points).
* Au(111) 3x3x4, vacuum 12 Å on each side (ASE `fcc111` convention, as upstream), tags 3–4 (bottom 2 layers) fixed.
* Hg starts 2.5 Å above the relaxed slab at ontop/bridge/fcc/hcp. Hg is fixed laterally (relaxes in z only), so each site stays at its high-symmetry position. An unconstrained relaxation is also run → `mlip_site_stability.csv`.
* BFGS, fmax = 0.02 eV/Å, ≤ 1000 steps, for every relaxation.
* Reference schemes: **A** Hg atom in a 15 Å box; **B** Hg fixed 10 Å above the top layer (fcc lateral position) in the same cell, slab relaxed; **C** (extra diagnostic) bulk α-Hg (full cell relaxation) + experimental cohesive energy 0.67 eV/atom.
* Rigid dissociation scan of Hg above the fcc site, 1.8–12 Å → `mlip_scan.csv`.
* CHGNet: weights 0.3.0, float32, CUDA. MACE: MACE-MP-0 medium (2023-12-03), float64, CUDA, no D3 add-on.
* LJ = ASE defaults, element-blind. Kept only as a numerical control (its E(atom) = 0, so A = B by construction) and excluded from the figures.

## 4. DFT reference (Quantum ESPRESSO)

All QE settings live in one place, `work/dft/qe_common.py` (pseudopotentials, smearing, thresholds, D3). Every run writes `pw.in`, `pw.out` and `run.json` (parsed results + the full input dict) under `work/dft/runs/<label>/`.

```bash
cd work/dft
../../venv/bin/python dft_timing.py cpu 2          # one SCF of Hg@fcc, 3x3x4 slab (timing)
../../venv/bin/python dft_convergence.py           # ~1.5 h -> results/dft_convergence.csv
../../venv/bin/python dft_production.py --plan     # show the job list
../../venv/bin/python dft_production.py --functional PBE      # resumable (finished runs are skipped)
../../venv/bin/python dft_production.py --functional PBE-D3   # starts from the PBE geometries
```
Production was run as two parallel workers on the same run tree: one on the CPU (`--engine cpu`, 10 MPI ranks) and one on the GPU (`--engine gpu`, `pw-gpu.x`, 1 rank; about 1.5× faster than the 10 CPU cores in a measured test, `work/dft/gpu_test.py`). A `LOCK` file in each run directory stops two workers from running the same calculation. In its first pass a worker skips sites another worker is running, then waits for them. `--sites` sets each worker's order. Both engines are QE 7.6 with identical inputs; `run.json` records which engine ran each calculation:
```bash
python dft_production.py --functional PBE --engine gpu --sites hcp fcc
python dft_production.py --functional PBE --engine cpu --sites ontop bridge
python dft_production.py --functional PBE-D3 --engine gpu --sites hcp fcc bridge ontop
python dft_production.py --functional PBE-D3 --engine cpu --sites ontop bridge fcc hcp
```
* Engines: QE 7.6 in two builds, `pw-mkl.x` (CPU, MKL, 10 MPI ranks, `-nk 2`) and `pw-gpu.x` (CUDA, 1 MPI rank, one RTX PRO 4000). In an early timing test the GPU was occupied by another job (out of memory with 3 GB free); once it was free, production used both engines in parallel (see above). `run.json` in each run directory records which engine ran that calculation.
* Pseudopotentials: SSSP 1.3 precision (md5-checked copies in `work/dft/pseudo/`).
* Production cutoff 50/300 Ry. Convergence test: E_ads changes < 0.3 meV up to 80/480 Ry. k 4×4×1 (error ≈ 10–15 meV from the k test). MV smearing 0.01 Ry, nspin = 1, conv_thr 1e-8 Ry, forc_conv_thr 7.78e-4 Ry/bohr (= 0.02 eV/Å).
* Fully fixed atoms (bottom 2 layers; Hg in scheme B) get their own species label (`Au1`, same pseudopotential and mass; `split_fixed_species` in `qe_common.py`). Without it QE keeps the inversion of the clean 4-layer slab, which maps fixed onto free layers, and the first BFGS step stops with `checkallsym`. The failed first attempt is kept in `work/dft/runs/failed/`.
* `upscale = 1` (&IONS): `conv_thr` stays at 1e-8 Ry throughout each relaxation. With QE's default (100), `conv_thr` is tightened to 1e-10 Ry near convergence. On the GPU build an SCF then stalled at about 2e-9 Ry for 200 iterations (archived in `work/dft/runs/failed/PBE_site_hcp_gpu_scf_stall`). The relaxations finished before this change (PBE slab, far, ontop, bridge) used the default; the effect on energies is < 1e-6 eV.
* PBE-D3 = `vdw_corr='dft-d3'`, `dftd3_version=4` (BJ damping), `dftd3_threebody=.true.`
* Measured cost (37 atoms, 696 electrons, 418 bands): one SCF from scratch ≈ 54 min at 65/390 Ry on a loaded machine; ≈ 15 min at 50/300 Ry with 10 free cores.

Outputs: `results/dft_results.csv`, `results/dft_bulk_refs.csv`, logs `results/dft_production_*.log`.

## 5. Experimental reference
`results/literature_experimental.md`: −ΔH_ads^Au(Hg) = 98 ± 3 kJ/mol (Soverna et al. 2005, as cited and used in Lens et al. 2018), with the lower limit 67 kJ/mol from Lens et al. 2018. Every DOI was checked through Crossref; the file states which values were read in full text.

## 6–7. Tables, figures, report
```bash
cd work
../venv/bin/python make_tables.py      # report/tables/*.tex + results/comparison_table.csv
../venv/bin/python plot_results.py     # report/figures/*.pdf|png
cd ../report && ../tools/tex/bin/tectonic main.tex   # local compile check (tectonic in tools/tex)
```
For Overleaf, upload `report/` (main.tex, refs.bib, figures/, tables/).

---

# MLIP survey for the HybriLIT talk (27 Nov 2026) — Stage 1

> **Not in this repository (survey):** `venvs/` (rebuild with `work/survey/install_envs.sh`), `upstream_latest/` (https://github.com/miroi/open-collection, commit 7d60709; the ALIGNN-FF script copy is in `work/survey/upstream_runs/alignn_ff/`, © Miro Iliaš, MIT), compiled NequIP/Allegro models (`nequip-compile`, see below), fine-tuned checkpoints (`work/survey/finetune/runs/`, retrain with the scripts), MPtrj replay subsets, `.traj` trajectories and `_discarded_*` runs. Paths are those of the original workstation.


```
venvs/<env>/             one venv per MLIP (mace chgnet matgl alignn sevenn mattersim orb fairchem grace nequip deepmd upet), venvs/_tools = uv
upstream_latest/         sparse clone of miroi/open-collection at 7d60709 (2026-11-03), contains ML-ALIGNN-FF; NOT modified
work/survey/             scripts of the survey
results/mlip_inventory.md  Stage 1 deliverable (inventory, install record, smoke test, Miro reproduction)
results/survey_logs/     install logs, pip freezes, smoke-test JSON/CSV, compile logs
results/hg_au/<model>/   Hg/Au(111) 411-protocol results per model
tools/cuda13             symlink to NVHPC 26.9 CUDA 13.3 (CUDA_HOME for nequip-compile)
```

```bash
bash work/survey/install_envs.sh                 # all envs (or: install_envs.sh mace orb ...), ~30 min with downloads
# NequIP/Allegro foundation models must be compiled once:
export CUDA_HOME=$PWD/tools/cuda13
for m in NequIP-OAM-L Allegro-OAM-L; do venvs/nequip/bin/nequip-compile nequip.net:mir-group/$m:0.1 \
    work/survey/models/$m.nequip.pt2 --device cuda --mode aotinductor --target ase; done
# smoke test of one model (env of the model: see VENV in work/survey/calcs.py)
cd work/survey && ../../venvs/mace/bin/python smoke_test.py MACE-MPA-0-medium
# Miro's ALIGNN-FF script, unchanged (CPU)
cd work/survey/upstream_runs/alignn_ff && ../../../../venvs/alignn/bin/python Hg_on_Gold111_adsorption.py
# site-label check and protocol bridge (Miro -> 411)
cd work/survey && ../../venvs/alignn/bin/python check_miro_sites.py
../../venvs/alignn/bin/python hg_au_protocol.py ALIGNN-FF-mptrj     # 411 protocol, ~20 s on the GPU
../../venvs/alignn/bin/python protocol_bridge.py ALIGNN-FF-mptrj    # needs hg_au_protocol output first
```
* `work/survey/calcs.py` is the only place where calculators are built (model, head/task, dtype, device,
  wrappers). Every script imports it, so settings cannot differ silently between models.
* `work/survey/hg_au_protocol.py` is generated from `work/mlip_benchmark.py`; `SETTINGS` is
  byte-identical to the 411 protocol (checked with diff) + scheme D (E_coh,PBE of Hg from `results/dft_bulk_refs.csv`).
* UMA (fairchem) needs Hugging Face access to `facebook/UMA`; not granted on this machine (403).

## MLIP survey — Stage 2 (Hg/Au(111), 31 model/task combinations)

```bash
cd work/survey
python3 run_all_hg_au.py                      # every model in its own venv, resumable; ~20 min on the GPU in total
python3 run_all_hg_au.py --models ORB-v3-cons-inf-omat --force   # rerun selected models
../../venvs/mace/bin/python collect_hg_au.py  # -> results/hg_au_all_models.csv, hg_au_model_summary.csv, figures/survey_*
```
Findings and caveats: `results/hg_au_stage2_notes.md`. UMA uses the fairchem `"batch"` preset (FP32, no
torch.compile); its venv uses uv's managed CPython 3.12 (ships Python.h, needed by triton in torch 2.13).

## MLIP survey — Stage 3 (published DFT: Steckel 2008, VASP-PBE, Hg on 6 metals)

```bash
cd work/survey
# 1. reference table (44 systems) parsed from the OSTI full text of Steckel, PRB 77, 115412 (2008)
pdftotext -layout steckel_osti.pdf steckel.txt && python3 -I build_steckel_table.py steckel.txt
# 2. all models (resumable), ~40 min on the GPU in total
python3 run_all_hg_au.py --task published
# 3. errors per source/subset, parity plots
../../venvs/mace/bin/python published_analysis.py
```
Outputs: `results/published/steckel2008_systems.csv` (sources), `results/published_benchmark.csv` (wide:
system, source, DOI, functional, DFT value, one column per model), `results/published_benchmark_long.csv`,
`results/published_mae.{csv,md}`, `results/figures/published_*.png|pdf`, notes and talk subset in
`results/published_notes.md`. Main MAE = 30 systems with 1 Hg per cell; the 14 systems with >1 Hg per cell
(per-Hg normalisation not stated in the paper) and the 5 Ni systems are reported separately.

## MLIP survey — Stage 4 (fine-tuning feasibility test: MACE-MP-0 medium, CHGNet 0.3.0)

```bash
cd work/survey/finetune
../../../venvs/mace/bin/python prepare_ft_data.py MACE-MP-0-medium      # splits + per-element energy shift
../../../venvs/chgnet/bin/python prepare_ft_data.py CHGNet-0.3.0
bash train_all.sh                      # 2 models x {noatom, atom}; MACE ~20 min/run, CHGNet ~3 min/run
bash eval_all.sh                       # frame errors, 411 protocol, Steckel test for the fine-tuned models
../../../venvs/mace/bin/python ft_analysis.py   # -> results/finetune/ft_summary.{csv,md}, figures/finetune_summary
```
Notes, results and pitfalls: `results/finetune/stage4_notes.md`. Fine-tuned models are registered in
`work/survey/calcs.py` (keys `*-ft-noatom`, `*-ft-atom`) and are excluded from the Stage 2/3 tables.
Stage 4b (lr vs replay, MACE): `bash work/survey/finetune/train_mace_ctrl.sh` (~3 min) and
`bash work/survey/finetune/train_mace_mh.sh` (replay, downloads mp_traj_combined.xyz 595 MB, ~50 min), then the same
evaluation commands with keys `MACE-MP-0-ft-atom-lr1e-4` / `MACE-MP-0-ft-atom-replay`; table
`results/finetune/ft_lr_replay_comparison.csv`. CHGNet isolated-atom reproduction: `work/survey/finetune/repro/`.

## MLIP survey — Stage 5 (report and talk)
* `report/mlip_survey.md`: recommendations by purpose, risks for heavy/superheavy elements, findings F1–F4.
* `report/talk_outline.md`: title proposals, 150-word abstract, 8-slide structure with the figure for each slide.
* Talk figures (all in `results/figures/`): `survey_atom_error_talk`, `element_coverage`
  (`work/survey/plot_element_coverage.py`), `published_parity_talk`, `finetune_lr_replay`
  (`work/survey/finetune/plot_lr_replay.py`), `finetune_summary`, `survey_sites`.
