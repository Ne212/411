# Stage 3: proposed reference data (awaiting approval, 2026-11-04)

| # | Source | Systems | DFT level | E_ads definition | Access | Proposal |
|---|---|---|---|---|---|---|
| 1 | Steckel, Phys. Rev. B 77, 115412 (2008), doi:10.1103/PhysRevB.77.115412 | Hg on Ag, Au, Cu, Ni, Pd, Pt: (111) at 0.083/0.25/0.333/0.5 ML (24, Table II); (001) at 0.125/0.5/1.0 ML (18, Table III); Au(001) bridge and top (2, Table I) → **44** | VASP-PAW, **PBE**, 380 eV, 4-layer slab, bottom 2 fixed, top 2 + Hg relaxed | per Hg vs isolated Hg atom | full text open (OSTI) | **main set**: same functional and code family as the MLIP training data |
| 2 | Pershina & Iliaš, Dalton Trans. 51, 7321 (2022), doi:10.1039/d2dt00240j; Pershina, Inorg. Chem. 57, 3948 (2018), doi:10.1021/acs.inorgchem.8b00101; Ryzhkov et al., PCCP 25, 15362 (2023), doi:10.1039/d3cp00856h; Iliaš & Pershina, Mol. Phys. (2024), doi:10.1080/00268976.2024.2363408; Pershina & Iliaš, Mol. Phys. (2025), doi:10.1080/00268976.2025.2573831 | M/Au(111), M = Hg, Tl, Pb, Bi (+ Po, At, Rn for models that accept them); superheavy elements documented as "outside every training set" | BAND (Slater-type orbitals), ZORA scalar-relativistic and 2c spin–orbit, functionals per paper | per paper (to be read) | **paywalled; full text not accessible here** → PDFs needed from Neşet / Miro | heavy-element set, ≈15–30 systems, only after the PDFs arrive |
| 3 | OC20 (Chanussot et al., ACS Catal. 11, 6059 (2021), doi:10.1021/acscatal.0c04525), Au-containing slabs in IS2RE val splits | ≈100–200 small-molecule adsorbates (O, H, OH, CO…) on Au and Au alloys | VASP, **RPBE** | vs gas-phase molecule references (OC20 convention) | 8.1 GB download (97 GB unpacked) + 208 MB mapping | **not recommended**: no metal adatoms; RPBE; part of the training data of 4 models |
| 4 | Catalysis-Hub | — | various (often BEEF-vdW) | per publication | API now needs a key (account) | not recommended unless Neşet wants it |

Element coverage test (`results/survey_logs/element_coverage.jsonl`): no model has superheavy elements
(Z ≥ 104) in its training data. ALIGNN-FF, SevenNet-Omni, ORB (except Og) and DPA-4 still *accept*
Cn…Og and return numbers from untrained embeddings. MACE, NequIP, GRACE, PET and TensorNet reject
Po, At and Rn as well.

Protocol for set 1 (each model): build the overlayer cell from the paper (4 layers, bottom 2 fixed, Hg in
the fcc hollow on (111) / 4-fold hollow on (001)) at the model's own a0. Then
(a) "source geometry": Hg placed at the paper's M–Hg distance, slab ideal, single point;
(b) "model geometry": top 2 layers + Hg relaxed, BFGS fmax 0.02 eV/Å.
Report both. Metrics: MAE and signed error vs PBE (absolute E_ads, scheme A as in the paper), the same after
the free-atom correction (scheme D), and reference-free differences (metal-to-metal and coverage trends
relative to Au(111) 0.25 ML). The 411 QE-PBE value (−0.333 eV, 0.111 ML) is a cross-check against Steckel
Au(111) (−0.36 eV at 0.083 ML, −0.38 at 0.25 ML).
Estimated cost: 31 models × 44 systems × 2 modes, < 1 h on the GPU.
