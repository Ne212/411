# HybriLIT Workshop, 27 November 2026 — 10-minute talk: outline

## Title (proposals; first one recommended)
1. **Universal machine-learning potentials for heavy-element adsorption on gold: the free-atom problem and what fine-tuning can fix**
2. Can off-the-shelf ML potentials replace DFT for Hg on gold? A 31-model benchmark and a fine-tuning test
3. From Hg to copernicium: how far do universal ML interatomic potentials go?

## Abstract (150 words)
Universal machine-learning interatomic potentials (MLIPs) promise DFT-quality adsorption energies at a fraction of
the cost. We tested 31 ASE-compatible models and tasks from twelve packages (MACE, CHGNet, MatGL, ALIGNN-FF,
SevenNet, MatterSim, ORB, UMA, GRACE, NequIP/Allegro, DeePMD, PET) for Hg on Au(111), against our Quantum
ESPRESSO PBE results and 30 published VASP-PBE systems on six metals. The best models reach errors of 0.03–0.04 eV,
but most models carry a large error in the energy of the free gas-phase atom (−0.61 to +0.45 eV). This error,
not the adsorbate–surface bond itself, dominates their adsorption energies. No model contains superheavy elements in its training
data. Fine-tuning MACE on 84 DFT frames plus the free atom removed the error on a held-out adsorption site. Naive training
erased other metals from the model; replay of foundation data prevented this. CHGNet cannot learn isolated atoms. Next:
fine-tuning on relativistic DFT data for superheavy elements on gold.

## Slides (8; ≈ 75 s each)
| # | Slide | Message | Figure (all in `results/figures/`) |
|---|---|---|---|
| 1 | Title | Why MLIPs for Hg/Cn on gold: DFT cost vs chemistry experiments | — (title slide) |
| 2 | 31 models, 12 packages | What exists, what is fine-tunable, what the training data are (PBE vs RPBE). Pitfalls in one line: gated UMA, MatGL stress unit, ORB/PET slabs | table from `results/mlip_inventory.md` (8-model subset, `results/talk_subset.csv`) |
| 3 | Same test for all: Hg/Au(111) | Models disagree with each other by more than 0.5 eV; protocol matters (Miro's script: +0.5 eV from the unrelaxed slab, site-label mix-up) | `survey_sites.png` (or a 4-site panel for the subset) |
| 4 | **Message 1: the free-atom error** | Every off-the-shelf MLIP has it (−0.61…+0.45 eV); it explains the E_ads deviations (r = −0.96); removing it drops the error to ≈ 0.05 eV; agreement with PBE-D3 can be pure error cancellation | `survey_atom_error_talk.png` (TensorNet hatched as the extra example) |
| 5 | Published data: Hg on six metals | Against Steckel 2008 (VASP-PBE), best 0.03–0.04 eV (NequIP, MACE-MH-1, UMA); OC20 heads +0.3–0.4 eV (RPBE); trends mostly right | `published_parity_talk.png` |
| 6 | **Message 2: superheavy elements are out of scope** | No model has Z ≥ 104 in its training data; some codes still return numbers (untrained embeddings, not shown) | `element_coverage.png` |
| 7 | **Message 3: fine-tuning works, forgetting is the risk** | MACE + free atom: held-out site −0.31 → −0.01 eV, but Ag/Cu/Ni/Pd/Pt lost; replay keeps them (free atom fixed, other metals 0.42 → 0.17 eV), at the price of a smaller gain on the site (−0.12 eV) in 15 epochs; CHGNet silently drops isolated atoms. One seed, one split, 84 frames: a feasibility test | `finetune_lr_replay.png` (+ `finetune_summary.png` as backup) |
| 8 | **Message 4: next step** | Train MACE with replay on Dip's relativistic QE data (Hg/Cn, Pb/Fl, Tl/Nh on Au); test on held-out sites, Pershina et al. 2c-DFT values and untouched elements | — (bullet slide; recommendation table from `report/mlip_survey.md` §4) |

Backup slides: Miro-script protocol bridge (`results/mlip_inventory.md` §5 table); UMA non-locality (Stage 2 notes);
ALIGNN PES discontinuity; CHGNet `dataset.py` lines 184–191 and the 3-line reproduction output
(`results/finetune/stage4_notes.md`).

## Wording to keep honest on the slides
* "fine-tuning test" = feasibility test: 84 frames, one QE study, one seed, one held-out site.
* Replay vs control differ also in the number of updates on our data (mace-torch triplicated it in the replay run).
* Superheavy: say "outside the training data", never show model numbers for Cn…Og.
* Steckel's per-Hg normalisation is not stated in the paper; the main MAE uses only 1-Hg cells.
