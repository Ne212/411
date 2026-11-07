# Stage 2: Hg/Au(111) with 31 model/task combinations (411 protocol)

Run 2026-11-04 on the RTX PRO 4000 (MatGL on CPU). The protocol and SETTINGS are identical to project 411
(`work/mlip_benchmark.py`): each model's own a0, 3×3×4 slab, bottom 2 layers fixed, Hg relaxed along z
only, BFGS fmax 0.02 eV/Å. Reference: QE PBE and PBE-D3 from project 411.

Files
* `results/hg_au_all_models.csv`: model × site × scheme (A, B, C, D), plus PBE and PBE-D3 rows; includes
  height, d(Au–Hg), convergence, largest Au displacement, training data and functional of the head used.
* `results/hg_au_model_summary.csv`: one row per model: a0, B0, E_coh(Hg), free-atom error, MAE vs PBE and
  PBE-D3 (schemes A and D), site differences, locality test, convergence.
* `results/figures/survey_sites.{png,pdf}`: E_ads at the four sites, scheme A vs scheme D.
* `results/figures/survey_atom_error.{png,pdf}`: free-atom (cohesive-energy) error per model.
* `results/figures/survey_site_differences.{png,pdf}`: E(site) − E(fcc), independent of the gas reference.
* per model: `results/hg_au/<model>/` (log, settings, structures, dissociation scan, site stability).

Schemes: A = isolated Hg atom in a 15 Å box (model energy); B = Hg fixed 10 Å above the slab, same cell;
C = bulk α-Hg(model) + experimental E_coh 0.67 eV; D = bulk α-Hg(model) + E_coh,PBE 0.169 eV (QE, 411).
"Free-atom error" = E_coh(model) − E_coh(PBE). It contains the error of the isolated atom *and* the error
of bulk α-Hg. The two cannot be separated, because the absolute energy zero of QE and of the VASP
training data differs.

## Findings (numbers from the CSVs above)

1. **The free-atom error dominates.** For the 26 PBE-type models (OC20 heads and the flagged AgAu head
   excluded), the free-atom error spans −0.61 (TensorNet-MatPES) to +0.45 eV (CHGNet), with median |error|
   0.12 eV. Its correlation with the deviation of E_ads(fcc, A) from PBE is r = −0.96.
2. **With the atom replaced by bulk Hg (scheme D), most models agree with PBE.** The median MAE over the four
   sites drops from 0.110 eV (A) to 0.048 eV (D). Best in D: NequIP-OAM-L 0.004, SevenNet-Omni-mpa 0.008,
   SevenNet-MF-ompa 0.011, DPA4-Plus 0.012, TensorNet-MatPES 0.016 eV.
3. **Good in A does not mean right.** SevenNet-MF-ompa (MAE vs PBE-D3 0.007 eV), ALIGNN-FF-omat24 (0.028)
   and M3GNet-MatPES (0.030) match PBE-D3 in scheme A, but none contains dispersion. Their free-atom
   errors (+0.34…+0.40 eV) happen to mimic the D3 binding (≈0.35 eV). This is error cancellation.
4. **Site differences (reference-free) are reproduced by almost all models.** PBE: ontop − fcc = 0.067,
   bridge − fcc = 0.018, hcp − fcc = 0.000 eV. Models give ontop − fcc = 0.07…0.32 eV. ALIGNN-FF (all three)
   puts bridge at or below fcc. The OC20 heads (MACE-MH-1, SevenNet-Omni, DPA-3.1, UMA) give an almost
   flat surface (ontop − fcc ≈ 0). Those heads are RPBE-type (see inventory), so their comparison with PBE
   is not like-for-like.
5. **Locality test.** E(Hg 10 Å above slab) − E(slab) − E(Hg atom) is 0.000 eV for all models except UMA
   (−0.13 eV for both tasks) and ORB (−0.003 / −0.008 eV); QE-PBE gives −0.0002 eV. For UMA, schemes A and
   B therefore differ by 0.13 eV. MAE vs PBE: UMA-omat 0.036 (A) / 0.098 (B) eV. Cause **not verified**.
   Possibly UMA's mixture-of-linear-experts coefficients depend on the global composition: the slab, the
   atom and slab+Hg have different compositions.
6. **Convergence and stability.** All relaxations (11 per model: slab, far reference, α-Hg cell, 4 sites with Hg on a z-line, 4 sites laterally free) converged except ALIGNN-FF-omat24 bridge, z-only and free (1000 BFGS
   steps, fmax 0.17 eV/Å, oscillating energy; the geometry is intact, largest Au displacement 0.06 Å).
   No surface reconstruction in any model (largest Au displacement 0.14 Å).
7. **Lattice constants.** PBE-type models give a0(Au) = 4.150–4.173 Å; QE-PBE: 4.150 Å.

## Checks of the pipeline
* CHGNet rerun with the new code reproduces project 411 to ≤ 0.1 meV.
* ALIGNN-FF-mptrj `protocol_bridge.py` step S4 equals the `hg_au_protocol.py` result to 1 meV.
* Full-pbc wrapper (ORB, PET) vs mixed pbc tested with MACE: identical energy.

## Not done / caveats
* UMA uses the fairchem "batch" preset (FP32, no compile) instead of the default "nitro" (TF32). Not compared.
* MatGL on CPU (GPU device mismatch in matgl 4.1.4).
* DPA-3.1-3M-AgAu-PBED3: in the CSVs, flagged; its dataset has no Hg.
* OC20 heads are compared with PBE although trained on RPBE (OC20 data); their MAEs mix model error and
  functional difference.
