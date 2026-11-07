# Stage 3: MLIPs vs published DFT (Hg on metal surfaces)

## Sources used and not used
* **Used, set 1:** Steckel, Phys. Rev. B 77, 115412 (2008), doi:10.1103/PhysRevB.77.115412. Full text from
  the public OSTI copy (OSTI 1010974). VASP-PAW, **PBE** (the main text states that all hollow-site data after
  Table I were generated with PBE; from Table I the PBE column is used). 44 systems, parsed from the PDF text by
  `work/survey/build_steckel_table.py` → `results/published/steckel2008_systems.csv` (with line numbers).
* **Separate source:** QE-PBE of project 411 (Hg/Au(111) fcc, 0.111 ML), from Stage 2. Reported on its own line.
* **Pending:** Pershina et al. (BAND, scalar-relativistic and 2c). Full texts not accessible here; Neşet is
  looking for them. Superheavy elements are not compared at all (no model has them in its training data; see the
  accept/reject table `results/survey_logs/element_coverage.jsonl`).
* **Not used:** OC20 and Catalysis-Hub. They contain mostly small-molecule adsorbates (O, H, OH, CO, …) rather
  than metal adatoms, and OC20 uses the RPBE functional (Catalysis-Hub now also requires an API key).

## Definitions and decisions
* E_ads per Hg. Scheme A: vs the isolated Hg atom of the same model (the paper's convention). Scheme D: the
  isolated atom is replaced by bulk α-Hg(model) + E_coh,PBE(QE, 0.169 eV), i.e. the model's free-atom
  (cohesive-energy) error is removed.
* **Normalisation:** neither the methods section nor the captions of Tables I–III say whether energies are per Hg
  or per cell. The paper only states that the 0.25 and 0.5 ML p(2×2) overlayers use the same supercell with
  one and two Hg atoms; the Hg count of the c(4×4) supercell is not given. The 14 systems with (possibly) >1 Hg
  per supercell are therefore **excluded from the main MAE** and reported separately. Main set: 30 systems.
  Indirect hint, not a verification: with per-Hg normalisation, the best models have the same small errors on
  those 14 systems (0.03–0.05 eV) as on the main set.
* **Ni** (5 of the 30): ferromagnetic; the paper's spin treatment is not stated, and the MLIPs have no spin
  input. Every table is also given without Ni (25 systems).
* Geometry: each model's own lattice constant. Mode "source" = ideal slab, Hg at the paper's M–Hg distance
  (single point). Mode "relaxed" = top 2 layers + Hg (z only) relaxed, as in the 411 protocol. Model Hg–Hg
  spacings match the paper's to ≤ 0.02 Å (overlayer cells verified). Source vs relaxed changes the MAE by
  0.005 eV (median) and 0.040 eV at most, so the relaxed values are quoted below.
* Unconverged relaxations (kept, flagged in `published_mae.csv`): ALIGNN-FF-mptrj Ni(001) ×3 (clean Ni(001)
  slab did not converge in 1000 steps); ALIGNN-FF-matpes-pbe Au(001) 0.125 ML.
  **Cause (short investigation, `work/survey/alignn_jump.py`, `alignn_neighbor_cap.py`, `alignn_discontinuity.py`):**
  the ALIGNN-FF-mptrj energy is discontinuous. On the clean Ni(001) slab, between two BFGS iterates (max atom
  move 0.06 Å), E jumps by 5.6 eV along the straight line between them (t = 0.7 → 0.8); the optimiser then
  oscillates between the two branches (log: −367.2 ↔ −361.7 eV) for 1000 steps. Ruled out: site slipping (the
  clean slab has no Hg; in the other runs Hg moves along z only), the max_neighbors = 52 cap (slab atoms have
  37–41 neighbours within 5 Å), and pair distances crossing 3, 4 or 5 Å (identical pair counts at t = 0.7 and 0.8).
  The root cause inside ALIGNN's graph construction or model is **not verified**. The Au(001) case shows
  similar energy jumps in its BFGS log (not analysed further). Stage 2's unconverged ALIGNN-FF-omat24 bridge
  run (oscillating energy) is probably the same phenomenon (not verified).

## Results (relaxed; eV). Full tables: `results/published_mae.md`, `results/published_mae.csv`
* **Steckel main set (30 systems, PBE).** For the 26 PBE-type models the median MAE is 0.135 eV in scheme A and
  0.070 eV in scheme D. Best: NequIP-OAM-L 0.034 (A) / 0.034 (D), MACE-MH-1-omat 0.039 / 0.038,
  UMA-s-1p2p1-omat 0.041 / 0.041, SevenNet-Omni-mpa 0.098 / 0.040. Worst: CHGNet-0.3.0 0.587 / 0.160,
  MACE-MP-0 0.414 / 0.219, ALIGNN-FF-mptrj 0.306 / 0.514 (Spearman 0.52: also wrong in trends).
* As in Stage 2, the free-atom error explains most of the large scheme-A errors: TensorNet-MatPES drops from
  0.637 (A) to 0.054 (D), GRACE-2L-OAM from 0.472 to 0.064, SevenNet-MF-ompa from 0.366 to 0.063.
* **OC20 heads** (MACE-MH-1-oc20, SevenNet-Omni-oc20, DPA-3.1-oc20m, UMA-oc20): mean signed error +0.32…+0.39 eV
  (A), i.e. weaker binding than PBE. This matches the RPBE training data of these heads. It is a functional
  difference, not necessarily a model error, so these heads are not ranked against the PBE models.
* **Ni:** MACE-MPA-0 overbinds Hg on Ni by 0.4–0.6 eV; without Ni its MAE is in the normal range.
* **QE-PBE 411 (separate):** NequIP-OAM-L −0.008 (A), SevenNet-Omni-mpa −0.128 (A) / −0.007 (D),
  MACE-MH-1-omat −0.034, UMA-omat −0.037, CHGNet −0.610, MACE-MP-0 −0.441, ALIGNN-FF-mptrj −0.320.
* The 411 QE-PBE value (−0.333 eV, d(Au–Hg) 3.007 Å, 0.111 ML) agrees with Steckel's VASP-PBE Au(111) values
  (−0.36 eV at 0.083 ML, −0.38 eV at 0.25 ML, d 2.99–3.01 Å) to within 0.05 eV.

## Talk subset (8 of 31), chosen by fixed rules (`work/survey/select_subset.py` → `results/talk_subset.csv`)
The rules were written down before they were applied, but **after** the Stage 2–3 results had been seen.
They are not a pre-registration.
R1 context (models of project 411 and of Miro's scripts) → CHGNet-0.3.0, MACE-MP-0, ALIGNN-FF-mptrj.
R2 user decision → UMA-s-1p2p1-omat in; TensorNet-MatPES-PBE out.
R3 every training-data group present (MPtrj; OMat24→sAlex+MPtrj; OMat24 only; OMat24 multi-domain; MatPES-PBE;
OC20 head).
R4 within a group, prefer an architecture class not yet in the subset; ties → lowest MAE (scheme D) on the
Steckel main set.
R4b for the OC20 group, prefer the OC20 head of a checkpoint already in the subset.
R5 all fine-tune support levels present (documented CLI / Python API only / continue-training only) → satisfied.
R6 at most 8 models.

| Model | Rule | Training data | Architecture | Fine-tuning | MAE_D Steckel |
|---|---|---|---|---|---|
| CHGNet-0.3.0 | R1 | MPtrj | invariant GNN, 3-body | Python API | 0.160 |
| MACE-MP-0-medium | R1 | MPtrj | MACE | documented CLI | 0.219 |
| ALIGNN-FF-mptrj | R1 | MPtrj | invariant GNN, 3-body | continue-training only | 0.514 |
| UMA-s-1p2p1-omat | R2 | OMat24 multi-domain | eSEN + mixture of linear experts | documented CLI | 0.041 |
| NequIP-OAM-L | R3+R4 | OMat24→sAlex+MPtrj | NequIP-type (e3nn) | documented CLI | 0.034 |
| DPA4-Plus-OMat24 | R3+R4 | OMat24 only | DeePMD DPA | documented CLI | 0.049 |
| M3GNet-MatPES-PBE | R3 (no new architecture in the group) | MatPES-PBE | invariant GNN, 3-body | documented CLI | 0.077 |
| UMA-s-1p2p1-oc20 | R3+R4b | OC20 head | eSEN + MoLE | documented CLI | 0.258 |

Compared with the first (hand-picked) proposal, the rules replace SevenNet-Omni-mpa, MACE-MH-1-omat and
MACE-MH-1-oc20 with DPA4-Plus-OMat24, M3GNet-MatPES-PBE and UMA-oc20.
TensorNet-MatPES-PBE appears **only** in the free-atom figure (`results/figures/survey_atom_error_talk`), hatched and
labelled "extra example", because it has the largest free-atom error of all 31 models (−0.61 eV) and still a small
error after the free-atom correction (0.054 eV), the clearest single illustration of the effect.
A first version of R4b compared architecture classes instead of checkpoints and wrongly picked MACE-MH-1-oc20; it
was corrected before the result above was used.

## Caveats
* One published source only (Steckel 2008, PBE). Its numerical precision is limited by the paper
  (2 decimals; reported convergence ±0.03 eV in layer count, ±0.02 eV in k-points).
* Scheme D uses the QE-PBE cohesive energy of Hg; it removes the free-atom error by construction, so it measures
  adsorption relative to bulk Hg, not the full model.
* No heavy-element (Tl, Pb, Bi…) data yet; that needs the Pershina papers.
