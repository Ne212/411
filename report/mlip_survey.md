# ASE-compatible machine-learning interatomic potentials for heavy-element adsorption on gold: survey, benchmark and fine-tuning test

Neşet Koçkar (Ankara University; JINR INTEREST project 411, with M. Iliaš and D. Sen, BLTP JINR). Work of 4 November 2026.
All numbers come from the files cited in brackets. How to reproduce everything: `README.md` (sections "MLIP survey").

## 1. Questions and short answers

| Question | Answer |
|---|---|
| Which MLIPs work with ASE and can be fine-tuned on our own DFT data? | All 12 packages tested (31 model/task combinations: MACE, CHGNet, MatGL, ALIGNN-FF, SevenNet, MatterSim, ORB, FAIRChem/UMA, GRACE, NequIP/Allegro, DeePMD/DPA, PET) run through ASE on the RTX PRO 4000 Blackwell. All have a fine-tuning route, from a documented CLI (MACE, SevenNet, NequIP, FAIRChem, GRACE, ORB, MatterSim, DeePMD, MatGL, metatrain) through a Python API (CHGNet) to "continue training" only (ALIGNN). [`results/mlip_inventory.md`] |
| How good are they for adsorption on metal surfaces, compared with published DFT? | Against VASP-PBE data for Hg on six metals (Steckel 2008, 30 systems), the best models reach a 0.03–0.04 eV MAE (NequIP-OAM-L, MACE-MH-1, UMA), the 26 PBE-type models a median of 0.135 eV. Most of the error comes from the energy of the free gas-phase atom, not from the adsorbate–surface bond. [`results/published_mae.md`] |
| Which MLIP for heavy and superheavy elements on gold? | For 6th-row elements on Au at the PBE level: NequIP-OAM-L, MACE-MH-1 or UMA, with an explicit check of the free-atom reference. **For superheavy elements, none**: no model has Z ≥ 104 in its training data. The route is fine-tuning on relativistic DFT data, where MACE (with replay) worked in our test. |

## 2. What was done
1. **Inventory** of 12 packages in separate venvs (CUDA 13 PyTorch for Blackwell). Recorded for each: training data, its
   functional, cutoff, licence, weight access and fine-tuning route. Every model was smoke-tested (energy, forces,
   finite-difference stress). [`results/mlip_inventory.md`]
2. **Miro's ALIGNN-FF script** was reproduced exactly (fcc −1.099, hcp −1.055, bridge −0.954, ontop −0.747 eV). Two
   findings: its "fcc" point is the hcp hollow and its "hcp" start point is not a hollow site. Changing one setting at a
   time shows that 0.47–0.52 eV of its adsorption energy comes from not relaxing the clean, 2.2 %-compressed slab.
   [`results/mlip_inventory.md` §5]
3. **Common Hg/Au(111) test** (project-411 protocol) on 31 model/task combinations, against our QE-PBE and PBE-D3.
   [`results/hg_au_stage2_notes.md`]
4. **Published-DFT benchmark:** Steckel, Phys. Rev. B 77, 115412 (2008), doi:10.1103/PhysRevB.77.115412 (VASP-PBE; Hg on
   Ag, Au, Cu, Ni, Pd, Pt (111)/(001)). 30 systems in the main set; 14 with unverified per-Hg normalisation and 5 Ni
   systems reported separately. OC20 and Catalysis-Hub were not used: they contain mostly small-molecule adsorbates
   rather than metal atoms, and OC20 uses the RPBE functional. [`results/published_notes.md`]
5. **Fine-tuning feasibility test** (MACE-MP-0, CHGNet) on 84 QE frames from project 411, with the ontop site held out,
   with and without the free Hg atom, plus a learning-rate-vs-replay comparison for MACE.
   [`results/finetune/stage4_notes.md`]

## 3. Main findings
**F1. Every off-the-shelf MLIP has a free-atom error.** The cohesive energy of Hg (free atom vs α-Hg) is wrong by
−0.61 to +0.45 eV across the 26 PBE-type models (median |error| 0.12 eV), and this error correlates with the
E_ads(fcc) deviation from QE-PBE with r = −0.96. An isolated atom has no neighbours within the cutoff, so the model
returns only a per-element reference term. That term was not fitted to a free-atom DFT energy; in MACE-MPA-0 and
MACE-OMAT-0 it is exactly the stored E0. Replacing the free atom by bulk α-Hg + the QE-PBE cohesive energy
(scheme D) brings the median four-site MAE vs QE-PBE from 0.110 to 0.048 eV, and the Steckel median MAE from 0.135
to 0.070 eV. Two corollaries:
* Some models agree with PBE-D3 only by error cancellation (SevenNet-MF-ompa: +0.34 eV free-atom error ≈ the D3 binding of
  0.35 eV); none contains dispersion.
* The "Hg far above the slab" reference cannot detect this error, because for a local model it is the free atom plus
  the slab. Exception: **UMA is not local** (E(Hg 10 Å above slab) − E(slab) − E(Hg atom) = −0.13 eV), so for UMA the
  two references differ.

**F2. Superheavy elements are out of scope.** The element tables of the MACE models trained on MPtrj, sAlex, OMat24
and OC20 end at Z = 94; CHGNet's embedding has 94 rows. ALIGNN-FF, SevenNet-Omni, ORB and DPA-4 nevertheless
*accept* Cn…Og and return numbers; these come from untrained embeddings and are not predictions (not shown
anywhere). MACE, NequIP, GRACE, PET and TensorNet even reject Po, At, Rn. [`results/figures/element_coverage.png`]

**F3. Fine-tuning can fix the problem (MACE), but forgetting is a real risk.** A single-seed, single-split
feasibility test on 84 frames shows:
* MACE-MP-0 + free Hg atom (lr 1e-3, 200 epochs): the held-out ontop E_ads error goes from −0.31 to −0.01 eV and the
  free-atom error from 0.22 to 0.00 eV. Au systems never seen in training (Steckel) improve from 0.37 to 0.04 eV.
  But Ag, Cu, Ni, Pd, Pt are **lost** (flat E(a)). A run with lr ≈ 0 reproduces the foundation model exactly, so
  the loss comes from training.
* With replay of 2000 MPtrj structures (lr 1e-4, 15 epochs), no metal is lost and the free-atom error is −0.002 eV.
  Steckel's other metals improve from 0.42 to 0.17 eV and drift by ≤ 0.026 Å (control without replay, same lr/epochs:
  0.30 eV, 0.050 Å). The held-out ontop error is −0.12 eV, not yet as small as in the aggressive run. Caveat: mace-torch
  triplicated our data in the replay run, so it made more updates on it than the control did.
* **CHGNet cannot learn the free atom** with its standard pipeline: `chgnet.data.dataset.StructureData`
  (`dataset.py` lines 184–191) silently replaces any structure that contains an isolated atom by a random other
  structure (minimal reproduction in `work/survey/finetune/repro/`). Its forces improve after fine-tuning
  (69 → 15 meV/Å), but its E_ads does not.

**F4. Other observations useful when choosing a model.**
* OC20 heads/tasks (MACE-MH-1, SevenNet-Omni, DPA-3.1, UMA) bind Hg 0.32–0.39 eV more weakly than PBE, consistent
  with their RPBE training data.
* ALIGNN-FF-mptrj has a discontinuous potential-energy surface: a 5.6 eV jump within 0.06 Å on Ni(001), which makes
  BFGS fail. The cause in the code is not verified.
* MatGL's ASE calculator returns stress in GPa by default (ASE needs eV/Å³).
* ORB and PET reject mixed-pbc slabs; UMA is gated on Hugging Face (manual approval).

## 4. Recommendations by purpose
| Purpose | Recommended | Why | Avoid / caution |
|---|---|---|---|
| Fast PBE-level screening of 6th-row adatoms (Hg, Tl, Pb, Bi) on Au and other metals | NequIP-OAM-L (MIT, fastest), MACE-MH-1-omat (ASL), UMA-s-1p2p1-omat (gated) | lowest error vs Steckel and QE-PBE (0.03–0.04 eV); free-atom error ≤ 0.02 eV | always report a bulk-reference check (scheme D) next to the free-atom reference; UMA: non-locality, check both references |
| Fine-tuning on our own (relativistic) DFT data | MACE (documented CLI; multihead replay keeps other elements) | worked in our test; free atom learnable | keep `--foundation_model_elements True`; use replay or low lr; include the free atom in the data |
| Fine-tuning, alternatives not tested here | SevenNet (reEWC forgetting-aware), NequIP (`nequip-train` from the package) | best/strong baselines with documented fine-tuning | test before relying on them |
| Comparison with RPBE data (OC20-type) | OC20 heads of UMA / MACE-MH-1 / SevenNet-Omni | trained on RPBE | not comparable with PBE numbers |
| Anything needing gas-phase references with CHGNet or ALIGNN-FF | — | CHGNet free-atom error 0.45 eV and not learnable; ALIGNN PES discontinuity | avoid for E_ads |
| Superheavy elements (Cn, Nh, Fl, …) | none off the shelf | no training data | numbers returned by accepting codes are meaningless |

## 5. Risks for heavy and superheavy elements
1. **Training-data coverage:** few compounds of the heavy 6p elements in MPtrj/OMat24, and none at all for Z ≥ 104.
   An accepted element is not a trained element.
2. **Relativistic effects:** the training data are scalar-relativistic PAW-PBE (VASP). Spin–orbit coupling, which
   matters for 6p and strongly for 7p (Nh, Fl, Mc) elements, is absent. A fine-tuned model can only be as good as
   its 2c/4c DFT training data.
3. **Free-atom reference:** not learned by foundation models (F1); must be in the fine-tuning data; CHGNet cannot
   take it.
4. **Forgetting:** narrow data can destroy other elements (F3); use replay and check bulk properties of untouched
   elements.
5. **Functional and dispersion mismatch:** PBE vs RPBE heads; none of the models contains dispersion, so a D3
   correction has to be added explicitly and reported.
6. **Method limits of this survey:** one published source (Steckel 2008), one QE study, single-seed fine-tuning.
   The Pershina et al. data (BAND, 2c) for Tl/Pb/Bi and their superheavy homologues were not accessible here.

## 6. Next step
Train/fine-tune MACE (with replay, free atoms included) on Dip's relativistic QE data for Hg/Cn, Pb/Fl, Tl/Nh on Au.
Test on held-out sites and on published 2c-DFT values (Pershina et al.), and check forgetting on bulk Au and on
untouched metals.

## References (identifiers checked via Crossref/DataCite/arXiv on 2026-11-04)
* Steckel, Phys. Rev. B 77, 115412 (2008), doi:10.1103/PhysRevB.77.115412 — benchmark data (VASP-PBE).
* Model papers and datasets: see `results/mlip_inventory.md` §6 (MACE-MP-0 arXiv:2401.00096; MACE-MH-1 arXiv:2510.25380;
  CHGNet doi:10.1038/s42256-023-00716-3; ALIGNN-FF doi:10.1039/D2DD00096B; SevenNet-Omni doi:10.1038/s41467-026-70195-8;
  UMA arXiv:2506.23971; NequIP foundation models arXiv:2607.28461; OMat24 arXiv:2410.12771; OC20 doi:10.1021/acscatal.0c04525).
* Heavy-element adsorption on gold, context only (no numbers used): Pershina & Iliaš, Dalton Trans. 51, 7321 (2022),
  doi:10.1039/d2dt00240j; Ryzhkov et al., Phys. Chem. Chem. Phys. 25, 15362 (2023), doi:10.1039/d3cp00856h;
  Pershina, Inorg. Chem. 57, 3948 (2018), doi:10.1021/acs.inorgchem.8b00101.
