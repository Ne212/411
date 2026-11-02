#!/usr/bin/env python
"""
Consistent MLIP benchmark: Hg adsorption on Au(111).

For every calculator (LJ control, CHGNet, MACE-MP-0 medium) the SAME protocol is used:
  1. Au fcc bulk lattice constant optimized with that calculator (Brent minimization of
     E(a) + Birch-Murnaghan fit on a +-3 % grid).
  2. Au(111) 3x3x4 slab built from that a, vacuum 12 A on each side, bottom 2 layers fixed.
  3. Hg at ontop / bridge / fcc / hcp (start height 2.5 A above the relaxed slab).
  4. E_ads with reference schemes
       A: E(slab+Hg) - E(slab) - E(Hg atom in 15 A cubic box)
       B: E(slab+Hg) - E(slab + Hg fixed 10 A above the top layer, same cell)
       C: E(slab+Hg) - E(slab) - E(bulk alpha-Hg per atom) - E_coh,exp(Hg)   [extra diagnostic]
  5. Rigid dissociation scan of Hg above the fcc site of the relaxed clean slab.

Outputs (in ../results):
  mlip_results.csv    one row per method x site x scheme
  mlip_bulk.csv       a0, B0 of Au; alpha-Hg bulk energy; Hg cohesive energy
  mlip_scan.csv       E(z) dissociation curves
  mlip_site_stability.csv  unconstrained (laterally free) relaxations per start site
  mlip_settings.json  every setting used
  structures/mlip/    relaxed structures (.traj and .xyz)

Usage:  python mlip_benchmark.py [--methods LJ CHGNet MACE]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111
from ase.constraints import FixAtoms, FixedLine
from ase.eos import EquationOfState
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
STRUCT = os.path.join(RES, "structures", "mlip")
os.makedirs(STRUCT, exist_ok=True)

# ---------------------------------------------------------------- settings
SETTINGS = dict(
    slab_size=(3, 3, 4),
    vacuum=12.0,                 # A, on each side (ASE fcc111 convention, as upstream)
    fixed_layers="tag >= 3 (bottom 2 of 4)",
    sites=["ontop", "bridge", "fcc", "hcp"],
    hg_start_height=2.5,         # A above top layer (as upstream)
    hg_lateral="fixed (FixedLine along z) so each site stays at its high-symmetry position; "
               "an unconstrained relaxation is also run per site -> mlip_site_stability.csv",
    far_height=10.0,             # A, scheme B
    far_lateral_site="fcc",
    atom_box=15.0,               # A, scheme A (as upstream)
    optimizer="BFGS",
    fmax=0.02,                   # eV/A, all relaxations, all methods
    max_steps=1000,
    eos_grid=dict(points=11, rel_range=0.03, eos="birchmurnaghan"),
    hg_bulk="alpha-Hg rhombohedral (ASE reference a=2.99 A, alpha=70.45 deg), "
            "full cell+position relaxation with FrechetCellFilter",
    Ecoh_exp_Hg=0.67,            # eV/atom, Kittel, Introduction to Solid State Physics, 8th ed., Table 3.1
    scan_heights=list(np.round(np.concatenate([np.arange(1.8, 4.0, 0.1),
                                               np.arange(4.0, 12.01, 0.25)]), 3)),
    scan_note="rigid scan: relaxed clean slab frozen, Hg above fcc hollow, single points",
)


# ---------------------------------------------------------------- calculators
def make_calc(name):
    if name == "LJ":
        from ase.calculators.lj import LennardJones
        # ASE defaults (epsilon=1 eV, sigma=1 A, rc=3 sigma) exactly as in upstream
        # LennardJones/Hg_on_Au-slab_LJ.py. Element-blind -> numerical control only.
        return LennardJones(), dict(model="ase.calculators.lj.LennardJones defaults",
                                    epsilon=1.0, sigma=1.0, rc=3.0, a_bracket=(1.2, 2.2))
    if name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        import chgnet
        calc = CHGNetCalculator(use_device="cuda", check_cuda_mem=False)
        return calc, dict(model="CHGNet 0.3.0 weights", package=f"chgnet {chgnet.__version__}",
                          device="cuda", dtype="float32", cutoff_A=6.0, a_bracket=(3.8, 4.4))
    if name == "MACE":
        from mace.calculators import mace_mp
        import mace
        calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
        return calc, dict(model="MACE-MP-0 medium (2023-12-03-mace-128-L1_epoch-199)",
                          package=f"mace-torch {mace.__version__}", device="cuda",
                          dtype="float64", dispersion=False, cutoff_A=6.0,
                          a_bracket=(3.8, 4.4))
    raise ValueError(name)


# ---------------------------------------------------------------- helpers
def energy(atoms, calc):
    atoms.calc = calc
    return atoms.get_potential_energy()


def relax(atoms, calc, label, log):
    atoms.calc = calc
    t0 = time.time()
    opt = BFGS(atoms, logfile=os.path.join(STRUCT, f"{label}.bfgs.log"),
               trajectory=os.path.join(STRUCT, f"{label}.traj"))
    converged = bool(opt.run(fmax=SETTINGS["fmax"], steps=SETTINGS["max_steps"]))
    f = atoms.get_forces(apply_constraint=True)
    fmax = float(np.sqrt((f ** 2).sum(axis=1)).max())
    e = atoms.get_potential_energy()
    write(os.path.join(STRUCT, f"{label}.xyz"), atoms)
    msg = (f"[{label}] steps={opt.nsteps} converged={converged} fmax={fmax:.4f} "
           f"E={e:.6f} eV ({time.time()-t0:.1f} s)")
    print(msg, flush=True)
    log.write(msg + "\n")
    if not converged:
        log.write(f"WARNING: {label} NOT converged\n")
    return e, opt.nsteps, converged, fmax


def optimize_au_lattice(calc, bracket):
    def e_of_a(a):
        return energy(bulk("Au", "fcc", a=a), calc)
    r = minimize_scalar(e_of_a, bounds=bracket, method="bounded",
                        options=dict(xatol=1e-5))
    g = SETTINGS["eos_grid"]
    a_grid = r.x * np.linspace(1 - g["rel_range"], 1 + g["rel_range"], g["points"])
    vols, ens = [], []
    for a in a_grid:
        at = bulk("Au", "fcc", a=a)
        vols.append(at.get_volume())
        ens.append(energy(at, calc))
    eos = EquationOfState(vols, ens, eos=g["eos"])
    v0, e0, B = eos.fit()
    a0 = (4 * v0) ** (1 / 3)
    return dict(a_brent=float(r.x), a0=float(a0), B0_GPa=float(B / 1e9 * 1.602176634e-19 * 1e30),
                E0_per_atom=float(e0))


def top_layer_z(atoms):
    tags = atoms.get_tags()
    return float(atoms.positions[(tags == 1), 2].mean())


def geom(atoms, i_hg, z_top):
    au = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Au"]
    d = atoms.get_distances(i_hg, au, mic=True)
    return float(atoms.positions[i_hg, 2] - z_top), float(d.min())


# ---------------------------------------------------------------- main protocol
def run_method(name, log):
    calc, info = make_calc(name)
    log.write(f"\n===== {name}: {json.dumps(info)}\n")
    print(f"\n===== {name}", flush=True)
    rows = []

    # 1. bulk lattice
    lat = optimize_au_lattice(calc, info["a_bracket"])
    a0 = lat["a0"]
    log.write(f"Au bulk: {lat}\n")
    print(f"Au bulk a0 = {a0:.4f} A, B0 = {lat['B0_GPa']:.1f} GPa", flush=True)

    # 2. clean slab
    def new_slab():
        s = fcc111("Au", size=SETTINGS["slab_size"], a=a0, vacuum=SETTINGS["vacuum"])
        s.set_constraint(FixAtoms(mask=s.get_tags() >= 3))
        return s

    slab = new_slab()
    e_slab, n, conv, fm = relax(slab, calc, f"{name}_slab", log)
    z_top_clean = top_layer_z(slab)

    # 3a. isolated atom (scheme A)
    hg = Atoms("Hg", pbc=True, cell=[SETTINGS["atom_box"]] * 3)
    hg.center()
    e_atom = energy(hg, calc)

    # 3b. far-away reference (scheme B): Hg fixed 10 A above top layer, slab relaxed
    far = slab.copy()
    add_adsorbate(far, "Hg", height=SETTINGS["far_height"],
                  position=SETTINGS["far_lateral_site"])
    i_far = len(far) - 1
    far.set_constraint(FixAtoms(mask=(far.get_tags() >= 3) | (np.arange(len(far)) == i_far)))
    e_far, n_far, conv_far, fm_far = relax(far, calc, f"{name}_far", log)
    h_far, d_far = geom(far, i_far, top_layer_z(far))

    # 3c. bulk alpha-Hg (scheme C diagnostic)
    hgb = bulk("Hg")
    hgb.calc = calc
    ecf = FrechetCellFilter(hgb)
    opt = BFGS(ecf, logfile=os.path.join(STRUCT, f"{name}_Hg_bulk.bfgs.log"))
    conv_hgb = bool(opt.run(fmax=SETTINGS["fmax"], steps=SETTINGS["max_steps"]))
    e_hg_bulk = hgb.get_potential_energy() / len(hgb)
    write(os.path.join(STRUCT, f"{name}_Hg_bulk.xyz"), hgb)
    ecoh = e_atom - e_hg_bulk
    cp = hgb.cell.cellpar()
    log.write(f"alpha-Hg: E/atom={e_hg_bulk:.6f} converged={conv_hgb} cellpar={cp.round(4).tolist()} "
              f"V/atom={hgb.get_volume()/len(hgb):.3f} E_coh(model)={ecoh:.4f}\n")
    print(f"E(Hg atom)={e_atom:.5f}  E(far)-E(slab)={e_far-e_slab:.5f}  "
          f"E_coh(Hg, model)={ecoh:.4f} eV", flush=True)

    bulk_row = dict(method=name, a0_A=a0, a_brent_A=lat["a_brent"], B0_GPa=lat["B0_GPa"],
                    E_slab=e_slab, slab_converged=conv, E_Hg_atom=e_atom,
                    E_far=e_far, far_converged=conv_far, far_height_A=h_far, far_dmin_AuHg_A=d_far,
                    E_far_minus_slab=e_far - e_slab,
                    E_Hg_bulk_per_atom=e_hg_bulk, Hg_bulk_converged=conv_hgb,
                    Hg_bulk_a_A=cp[0], Hg_bulk_alpha_deg=cp[3],
                    Hg_bulk_V_per_atom=hgb.get_volume() / len(hgb),
                    Ecoh_Hg_model=ecoh, Ecoh_Hg_exp=SETTINGS["Ecoh_exp_Hg"],
                    model=info["model"])

    # 4. adsorption at the four sites
    stab = []
    for site in SETTINGS["sites"]:
        s = slab.copy()
        add_adsorbate(s, "Hg", height=SETTINGS["hg_start_height"], position=site)
        i_hg = len(s) - 1
        xy0 = s.positions[i_hg, :2].copy()
        free = s.copy()      # unconstrained-laterally copy for the site-stability check
        s.set_constraint([FixAtoms(mask=s.get_tags() >= 3), FixedLine(i_hg, [0, 0, 1])])
        e_tot, nst, cv, fm = relax(s, calc, f"{name}_{site}", log)
        h, dmin = geom(s, i_hg, top_layer_z(s))
        drift = float(np.linalg.norm(s.positions[i_hg, :2] - xy0))
        free.set_constraint(FixAtoms(mask=free.get_tags() >= 3))
        e_free, nst_f, cv_f, fm_f = relax(free, calc, f"{name}_{site}_free", log)
        h_f, d_f = geom(free, i_hg, top_layer_z(free))
        stab.append(dict(method=name, start_site=site, E_constrained=e_tot, E_free=e_free,
                         dE_free_minus_constrained=e_free - e_tot,
                         lateral_drift_free_A=float(np.linalg.norm(free.positions[i_hg, :2] - xy0)),
                         height_free_A=h_f, d_AuHg_min_free_A=d_f, converged_free=cv_f,
                         n_steps_free=nst_f))
        refs = {"A_isolated_atom": e_slab + e_atom,
                "B_far_above_slab": e_far,
                "C_bulkHg_plus_Ecoh_exp": e_slab + e_hg_bulk + SETTINGS["Ecoh_exp_Hg"]}
        for scheme, eref in refs.items():
            rows.append(dict(method=name, a0_A=a0, site=site, scheme=scheme,
                             E_ads_eV=e_tot - eref, E_total_eV=e_tot, E_ref_eV=eref,
                             height_A=h, d_AuHg_min_A=dmin, lateral_drift_A=drift,
                             n_steps=nst, converged=cv, fmax_final=fm,
                             fmax_target=SETTINGS["fmax"]))
        print(f"  {site:6s} E_ads(A)={e_tot-refs['A_isolated_atom']:+.4f} "
              f"E_ads(B)={e_tot-refs['B_far_above_slab']:+.4f} "
              f"E_ads(C)={e_tot-refs['C_bulkHg_plus_Ecoh_exp']:+.4f}  h={h:.3f} d={dmin:.3f} "
              f"drift={drift:.3f}", flush=True)

    # 5. rigid dissociation scan
    scan = []
    for z in SETTINGS["scan_heights"]:
        s = slab.copy()
        add_adsorbate(s, "Hg", height=0.0, position="fcc")
        s.positions[-1, 2] = z_top_clean + z
        e = energy(s, calc)
        scan.append(dict(method=name, height_A=float(z), E_minus_slab_minus_atom=e - e_slab - e_atom))

    return rows, bulk_row, scan, stab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["LJ", "CHGNet", "MACE"])
    args = ap.parse_args()

    import ase, torch
    meta = dict(settings=SETTINGS, ase=ase.__version__, torch=torch.__version__,
                cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
    rows, bulks, scans, stabs = [], [], [], []
    with open(os.path.join(RES, "mlip_log.txt"), "w") as log:
        log.write(json.dumps(meta, indent=1, default=str) + "\n")
        for m in args.methods:
            r, b, s, st = run_method(m, log)
            rows += r; bulks.append(b); scans += s; stabs += st
    pd.DataFrame(rows).to_csv(os.path.join(RES, "mlip_results.csv"), index=False, float_format="%.6f")
    pd.DataFrame(bulks).to_csv(os.path.join(RES, "mlip_bulk.csv"), index=False, float_format="%.6f")
    pd.DataFrame(stabs).to_csv(os.path.join(RES, "mlip_site_stability.csv"), index=False,
                               float_format="%.6f")
    pd.DataFrame(scans).to_csv(os.path.join(RES, "mlip_scan.csv"), index=False, float_format="%.6f")
    with open(os.path.join(RES, "mlip_settings.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("done")


if __name__ == "__main__":
    main()
