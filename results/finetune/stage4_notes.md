# Stage 4: fine-tuning feasibility test (MACE-MP-0 medium, CHGNet 0.3.0)

**This is a feasibility test on a very small data set**: 84 training frames (+8 validation) from one QE-PBE
study (project 411: one Au(111) 3×3×4 slab, Hg at three sites, α-Hg, bulk-Au E(V)). It shows what fine-tuning
does and where it fails. It is not a production-quality fine-tune; nothing here should be read as a recommended
protocol or as a converged accuracy.

## Data and split (`work/survey/finetune/prepare_ft_data.py`, `results/finetune/` and `.../data/<model>/prep.json`)
* Source: `work/dft/runs/prod/PBE` only (one consistent setting, 50/300 Ry). PBE-D3 runs excluded (dispersion,
  not the foundation models' level); convergence-test runs excluded (other cutoffs/k-points). No Steckel data.
* Split by calculation type: **test = the whole ontop-site relaxation (16 frames), never trained on**;
  train/valid (84/8, random seed 0) from the clean slab, bridge, fcc, hcp relaxations, α-Hg vc-relax and bulk-Au
  E(V). Variant "no free atom" vs "+ free atom": the latter adds the isolated Hg atom and the 'far' frame (Hg 10 Å
  above the slab). 'far' is left out of the first variant because, for a local model, it carries the
  free-atom energy.
* Energy zero: QE energies shifted per element (least squares to the foundation model on the condensed training
  frames only). Same shift in both variants.

## Training (identical where the codes allow)
Full fine-tuning (no frozen layers, no replay), Adam, lr 1e-3, 200 epochs, batch 4, force:energy loss weight
10:1, seed 0. MACE: `mace_run_train --foundation_model … --multiheads_finetuning False --E0s foundation
--foundation_model_elements True`, float64, EMA 0.99, checkpoint with the lowest validation loss. CHGNet:
`Trainer` (energy per atom + forces, MSE, CosLR), checkpoint `bestE_*` (lowest validation energy MAE).
Loss definitions differ between the two codes, so the runs are comparable only qualitatively.

## Results (`results/finetune/ft_summary.md|csv`, `results/figures/finetune_summary.png|pdf`)
| | MACE found. | MACE ft no atom | MACE ft + atom | CHGNet found. | CHGNet ft no atom | CHGNet ft "+ atom" |
|---|---|---|---|---|---|---|
| E_ads ontop (held out) − QE, eV | −0.308 | −0.072 | **−0.014** | −0.468 | −0.518 | −0.509 |
| E_ads bridge/fcc/hcp, MAE vs QE, eV | 0.437 | 0.111 | **0.051** | 0.602 | 0.590 | 0.598 |
| free-atom error, eV | 0.219 | 0.104 | **−0.001** | 0.453 | 0.417 | 0.436 |
| held-out frames, force MAE, meV/Å | 55 | 7.7 | 7.5 | 69 | 14.6 | 15.3 |
| a0(Au) − QE, Å | +0.026 | +0.002 | +0.002 | +0.020 | −0.005 | −0.005 |
| B0(Au) − QE, GPa | −8.5 | −20.4 | −22.4 | −13.0 | +12.1 | +15.6 |
| Steckel Au rows (5, never trained), MAE A, eV | 0.372 | 0.066 | **0.041** | 0.527 | 0.548 | 0.569 |
| Steckel Ag/Cu/Ni/Pd/Pt (25), MAE A, eV | 0.423 | **lost** | **lost** | 0.599 | 0.750 | 0.686 |
| max \|Δa0\| of Ag/Cu/Ni/Pd/Pt vs foundation, Å | 0 | lost | lost | 0 | 0.059 | 0.062 |

1. **Memorised or generalised?** MACE generalises to the held-out ontop site: its error falls from −0.31 to
   −0.07 eV (no atom) and to −0.01 eV (+ atom). This is as good as on the trained sites. It also transfers to
   Au systems never seen in training (Steckel Au(111)/(001) at other coverages: 0.37 → 0.04 eV). Caveat: ontop
   is held out, but its Au environment (the same slab) is in the training data.
2. **The free atom must be in the data.** With MACE, adding the isolated Hg atom (+ 'far') removes the free-atom
   error (0.22 → 0.00 eV) and halves the remaining site error. Without it, the error drops only partly
   (0.22 → 0.10 eV), probably via the Hg self-energy term learned from the condensed frames (not verified).
3. **CHGNet cannot learn the free atom with this pipeline.** `chgnet.data.dataset.StructureData` silently omits
   structures that contain an isolated atom (no neighbour within the 6 Å graph cutoff). It replaces them by a
   random other structure (`dataset.py`, "Omit structures with isolated atoms"), with no log message. The
   free Hg atom and the 'far' frame are therefore never trained on, and the "+ atom" run equals the "no atom"
   run. Forces improve (69 → 15 meV/Å), but E_ads does not (error stays −0.5…−0.6 eV), because the error is in
   the gas-phase reference. Fixing it would need e.g. an Hg₂ dimer just inside the cutoff, i.e. new DFT, which
   was excluded.
4. **Catastrophic forgetting.** After naive full fine-tuning on Au/Hg-only data, MACE gives a **flat E(a)** for Ag,
   Cu, Ni, Pd, Pt (energy independent of the lattice constant within 1 meV/atom): bonding in these metals is lost.
   Check: the same command with lr = 1e-12 (1 epoch) reproduces the foundation curves exactly, so the loss is
   caused by training, not by model loading. CHGNet forgets partially: its Steckel error for the other metals
   rises from 0.60 to 0.69–0.75 eV, and their lattice constants move by up to 0.06 Å. Even Au's bulk modulus gets
   worse in both models (MACE −20 GPa, CHGNet +12…+16 GPa vs QE), although a0(Au) improves.
   Mitigations not tested here (no time / scope): MACE multihead fine-tuning with replay of foundation data
   (default in mace-torch, downloads replay data); SevenNet's forgetting-aware reEWC; freezing layers; lower lr
   or fewer epochs; adding other-metal data.

## Stage 4b: learning rate vs replay (MACE-MP-0, same data incl. free Hg atom; ONE seed, ONE split)
(a) main run: no replay, lr 1e-3, 200 epochs. (b) control: no replay, lr 1e-4, 15 epochs.
(c) multihead fine-tuning with replay of 2000 random MPtrj structures (`mp_traj_combined.xyz`, 595 MB, from
github.com/ACEsuit/mace-foundations), lr 1e-4, 15 epochs, evaluated with the fine-tuned head `Default`.
Scripts: `train_mace_ctrl.sh`, `train_mace_mh.sh`. Table: `results/finetune/ft_lr_replay_comparison.csv`.
A first replay run with lr 1e-3 was stopped after 2 epochs (≈3 min per epoch, 30 epochs would exceed 1 h, and
the replay-head validation error rose from 89 to 203 meV/atom). It is archived as `_discarded_*`.

| | foundation | (a) lr 1e-3, 200 ep | (b) lr 1e-4, 15 ep | (c) replay, lr 1e-4, 15 ep |
|---|---|---|---|---|
| E_ads ontop (held out) − QE, eV | −0.308 | **−0.014** | −0.256 | −0.124 |
| E_ads bridge/fcc/hcp MAE vs QE, eV | 0.437 | **0.051** | 0.322 | 0.189 |
| free-atom error, eV | 0.219 | −0.001 | 0.146 | **−0.002** |
| a0(Au) − QE, Å | +0.026 | +0.002 | −0.014 | −0.013 |
| B0(Au) − QE, GPa | −8.5 | −22.4 | +1.3 | −7.1 |
| Steckel Au rows (5), MAE A, eV | 0.372 | **0.041** | 0.240 | 0.106 |
| Steckel Ag/Cu/Ni/Pd/Pt (25), MAE A, eV | 0.423 | lost | 0.302 | **0.173** |
| Steckel Ag/Cu/Ni/Pd/Pt (25), MAE D, eV | 0.232 | lost | 0.210 | **0.174** |
| max \|Δa0\| Ag/Cu/Ni/Pd/Pt vs foundation, Å | 0 | lost (flat E(a)) | 0.050 | **0.026** |
| metals lost | none | Ag, Cu, Ni, Pd, Pt | none | none |

Reading (single seed, single split, one data set: indicative only):
* The total loss of the other metals in (a) is avoided by **both** lower lr / fewer epochs (b) and replay (c).
  So the collapse in (a) is mainly a consequence of the aggressive schedule.
* Replay is better than the control on every forgetting metric. Lattice drift of the other metals is 0.026 vs
  0.050 Å, the Steckel error on other metals 0.17 vs 0.30 eV, and the Au bulk modulus stays near the foundation
  value. This holds although (c) made **more** updates on our data than (b).
* Confounder: mace-torch triplicated our training data in (c) because the real/replay ratio (0.048) was below
  `real_pt_data_ratio_threshold` = 0.1 (log: "increasing the number of configurations…"). (c) therefore saw our
  frames 3× per epoch, (b) 1×. Part of (c)'s better fit (free atom −0.002 vs 0.146 eV; ontop −0.12 vs −0.26 eV)
  may come from that, not from replay itself (not verified).
* Neither (b) nor (c) reaches the adsorption accuracy of (a) within 15 epochs (ontop −0.12 eV vs −0.01 eV).
  Longer replay training was not run (≈3 min per epoch here).
* The replay head itself stayed close to the foundation model (validation 89 → 95 meV/atom, 117 → 108 meV/Å).
  In the stopped lr 1e-3 attempt it degraded to 203 meV/atom after 2 epochs.

## Verification of the CHGNet finding (for the talk)
Installed version: chgnet 0.4.2. File:
`venvs/chgnet/lib/python3.11/site-packages/chgnet/data/dataset.py`, class `StructureData`, `__getitem__`:
* lines 162–164: the graph is built with `self.graph_converter(struct, ...)`;
* lines 184–191: `except Exception:` → the structure is recorded in `self.failed_graph_id` / `self.failed_idx`, and
  `idx = random.randint(0, len(self) - 1)`; `return self.__getitem__(idx)`, i.e. a random other structure is
  returned instead (comment in the source: "Omit structures with isolated atoms. Return another randomly selected
  structure"). No message is printed.
* The exception comes from `venvs/chgnet/lib/python3.11/site-packages/chgnet/graph/converter.py`: lines 161–172.
  If an atom has no neighbour within `atom_graph_cutoff` (6 Å) and `on_isolated_atoms == "error"` (the default,
  line 43), then `raise ValueError(...)`.

Minimal reproduction: `work/survey/finetune/repro/chgnet_isolated_atom.py` (output in `chgnet_isolated_atom.out`):
```
ds[0] requested  Au4 -> returned graph with Au4 (Z=[79]), target e = -3.20
ds[1] requested  Hg1 -> returned graph with Au4 (Z=[79]), target e = -3.20
failed_graph_id: {np.int64(1): 'Hg1'} | failed_idx: [1]
ds2[1] requested Hg1 Au36 -> returned graph with Au4, target e = -3.20; failed_graph_id: {np.int64(1): 'Hg1 Au36'}
```
Asking for the free Hg atom (and for the 'far' Au36Hg slab) returns the Au4 graph with the Au4 energy.

MACE does not have this problem in our runs. mace-torch removes single-atom configurations only if
`config_type == "IsolatedAtom"` (`mace/data/utils.py`, lines 325–344, unless `--keep_isolated_atoms`). Ours is
labelled `Hg_atom`, and the training log reports 86 training configurations (84 + free atom + 'far').

## How easy was it? (honest notes)
| | MACE (mace-torch 0.3.17) | CHGNet (0.4.2) |
|---|---|---|
| Code needed | one CLI command (`train_all.sh`) | ~50-line Python script with the `Trainer` API |
| Wall time here | ~18–20 min per run (all 89 elements kept; GPU shared at times) | 3 min (10 min while sharing the GPU) |
| Documentation | README section + Colab; many options | README + example notebook |
| Pitfalls met | (a) by default the fine-tuned model keeps **only the training elements** (Au, Hg) and cannot evaluate Ag, Cu, …; `--foundation_model_elements True` is needed (`--foundation_filter_elements` is a misleadingly named, unrelated option); (b) E0 options (`foundation`/`estimated`/`average`) must match how the energies are referenced; (c) naive mode → forgetting | (a) isolated-atom structures silently dropped; (b) targets are energy **per atom**; (c) `Trainer.best_model` is a reference to the live model, not a copy, so use the `bestE_*`/`bestF_*` files; (d) foundation trained on MP2020-corrected energies, so a per-element shift (AtomRef refit) is needed for other DFT data |
| Discarded runs (kept, `_discarded_*`) | first MACE runs without `--foundation_model_elements` (element-filtered model) | — |

## Not done / limits
* One held-out split (ontop) only; no repeat with other seeds or other held-out sites; no uncertainty estimate.
* Scheme D and dispersion not used here. No D3 data. The Steckel test uses the fine-tuned models' own lattice
  constants (as in Stage 3).
* Dip's QE data are not included yet; the same scripts accept more `pw.out` directories.
