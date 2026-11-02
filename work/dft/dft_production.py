#!/usr/bin/env python
"""
DFT (Quantum ESPRESSO) reference for Hg on Au(111), same protocol as the MLIP benchmark.

Per functional (PBE, PBE-D3(BJ)):
  1. Au fcc a0 (Birch-Murnaghan, 1-atom cell, k 16^3)
  2. Au(111) 3x3x4 slab from that a0, vacuum 12 A each side, bottom 2 layers fixed, QE BFGS relax
  3. Hg at ontop/bridge/fcc/hcp, Hg relaxes along z only (if_pos 0 0 1),
     start height = MACE-relaxed height for that site
  4. references: A Hg atom in 15 A box (Gamma); B Hg fixed 10 A above top layer (fcc lateral),
     slab relaxed; alpha-Hg bulk (vc-relax) for the cohesive-energy diagnostic
QE's BFGS is used (not ASE's) so the wavefunctions and charge density are reused between ionic steps.
forc_conv_thr = 0.02 eV/A (per component), the same threshold as the MLIP runs (norm).

Usage:
  python dft_production.py --plan                # list jobs, nothing is run
  python dft_production.py --functional PBE      # run (resumable: finished jobs are skipped)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111
from ase.constraints import FixAtoms, FixCartesian
from ase.eos import EquationOfState
from ase.io import read

from qe_common import run_pw

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")

PROD = dict(
    cutoff={"system": {"ecutwfc": 50.0, "ecutrho": 300.0}},  # converged to 0.2 meV in E_ads (see dft_convergence.csv)
    kslab=(4, 4, 1),
    kbulk=(16, 16, 16),
    khg=(12, 12, 12),
    vacuum=12.0,
    sites=["ontop", "bridge", "fcc", "hcp"],
    far_height=10.0,
    atom_box=15.0,
    npool=2,
    engine="cpu",            # set from --engine; "gpu" -> pw-gpu.x, 1 rank, npool 1
)


def mace_heights():
    d = pd.read_csv(os.path.join(RES, "mlip_results.csv"))
    d = d[(d.method == "MACE") & (d.scheme == "A_isolated_atom")]
    return dict(zip(d.site, d.height_A))


def seed_from_pbe(func):
    """For PBE-D3: displacements of the relaxed PBE slab from its ideal positions + PBE Hg heights."""
    if func != "PBE-D3":
        return None
    out = os.path.join(HERE, "runs", "prod", "PBE", "slab", "pw.out")
    refs_csv = os.path.join(RES, "dft_bulk_refs.csv")
    while not (os.path.exists(out) and "JOB DONE" in open(out, errors="ignore").read()
               and os.path.exists(refs_csv) and "PBE" in set(pd.read_csv(refs_csv).functional)):
        print("waiting for the PBE slab / PBE reference data ...", flush=True)
        time.sleep(120)
    refs = pd.read_csv(refs_csv).set_index("functional")
    ideal = fcc111("Au", size=(3, 3, 4), a=refs.loc["PBE", "a0_A"], vacuum=PROD["vacuum"])
    rel = read(out, index=-1, format="espresso-out")
    return dict(slab_disp=rel.positions - ideal.positions)


def pbe_height(site):
    """PBE-relaxed Hg height for `site`; waits until another worker has written it."""
    csv = os.path.join(RES, "dft_results.csv")
    while True:
        if os.path.exists(csv):
            d = pd.read_csv(csv)
            d = d[(d.method == "PBE") & (d.scheme == "A_isolated_atom") & (d.site == site)]
            if len(d):
                return float(d.height_A.iloc[0])
        print(f"waiting for the PBE result of site {site} ...", flush=True)
        time.sleep(120)


def au_a0(func):
    vols, ens = [], []
    for a in np.linspace(4.00, 4.30, 9) if func == "PBE" else np.linspace(3.95, 4.25, 9):
        r = run_pw(bulk("Au", "fcc", a=a), f"prod/{func}/bulkAu_a{a:.4f}", func,
                   kpts=PROD["kbulk"], overrides=PROD["cutoff"], engine=PROD["engine"],
                   npool=5 if PROD["engine"] == "cpu" else 1)
        vols.append(r["atoms"].get_volume()); ens.append(r["energy_eV"])
    v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
    return (4 * v0) ** (1 / 3), B * 160.21766208


def relax(atoms, label, func, calc="relax", kpts=None, extra=None, on_busy="wait"):
    ov = {k: dict(v) for k, v in PROD["cutoff"].items()}
    ov["control"] = {"nstep": 200}
    for k, v in (extra or {}).items():
        ov.setdefault(k, {}).update(v)
    r = run_pw(atoms, f"prod/{func}/{label}", func, calculation=calc,
               kpts=kpts or PROD["kslab"], overrides=ov, engine=PROD["engine"],
               npool=PROD["npool"] if PROD["engine"] == "cpu" else 1, on_busy=on_busy)
    if r is None:
        print(f"[{func}/{label}] running in another worker -> skipped for now", flush=True)
        return None, "BUSY"
    flag = "OK" if (r["job_done"] and (calc == "scf" or r["relax_converged"])) else "NOT CONVERGED/FAILED"
    print(f"[{func}/{label}] E={r['energy_eV']} n_scf={r['n_scf']} {flag} {r['wall_pw']}", flush=True)
    if flag != "OK" or r["atoms"] is None or r.get("scf_not_converged"):
        raise RuntimeError(f"{func}/{label} did not converge - stopping this worker for inspection")
    return r, flag


def top_z(atoms, n_top=9):
    z = np.sort(atoms.positions[[a.index for a in atoms if a.symbol == "Au"], 2])
    return z[-n_top:].mean()


def run(func):
    rows, info = [], {}
    a0, B0 = au_a0(func)
    info.update(functional=func, a0_A=a0, B0_GPa=B0)
    print(f"{func}: a0={a0:.4f} A B0={B0:.1f} GPa", flush=True)

    slab = fcc111("Au", size=(3, 3, 4), a=a0, vacuum=PROD["vacuum"])
    tags = slab.get_tags().copy()
    seed = seed_from_pbe(func)
    start = slab.copy()
    if seed is not None:   # PBE-D3: transfer the PBE relaxation displacements onto the D3 lattice
        start.positions += seed["slab_disp"]
        info["start_geometry"] = "ideal D3 slab + PBE relaxation displacements; Hg at PBE heights"
    else:
        info["start_geometry"] = "ideal slab; Hg at MACE-relaxed heights"
    start.set_constraint(FixAtoms(mask=tags >= 3))
    r_slab, f_slab = relax(start, "slab", func)
    slab_rel = r_slab["atoms"]; slab_rel.set_tags(tags)
    e_slab = r_slab["energy_eV"]

    hg = Atoms("Hg", pbc=True, cell=[PROD["atom_box"]] * 3); hg.center()
    r_atom = run_pw(hg, f"prod/{func}/Hg_atom", func, kpts=None, overrides=PROD["cutoff"],
                    engine=PROD["engine"])
    e_atom = r_atom["energy_eV"]

    hgb = bulk("Hg")
    r_hgb, f_hgb = relax(hgb, "Hg_bulk", func, calc="vc-relax", kpts=PROD["khg"],
                         extra={"control": {"tstress": True}})
    e_hgb = r_hgb["energy_eV"] / len(hgb)

    far = slab_rel.copy()
    far.info["adsorbate_info"] = slab.info["adsorbate_info"]
    add_adsorbate(far, "Hg", height=PROD["far_height"], position="fcc")
    far.positions[-1, 2] = top_z(slab_rel) + PROD["far_height"]
    far.set_constraint(FixAtoms(mask=np.append(tags >= 3, True)))
    r_far, f_far = relax(far, "far", func)
    e_far = r_far["energy_eV"]

    info.update(E_slab=e_slab, slab_status=f_slab, E_Hg_atom=e_atom, E_Hg_bulk_per_atom=e_hgb,
                Hg_bulk_status=f_hgb, Ecoh_Hg=e_atom - e_hgb, E_far=e_far, far_status=f_far,
                E_far_minus_slab_minus_atom=e_far - e_slab - e_atom)

    h_mace = mace_heights()
    h_start = (lambda st: pbe_height(st)) if seed is not None else (lambda st: h_mace[st])
    pending = list(PROD["sites"])
    for on_busy in ("skip", "wait"):       # first pass skips sites another worker is running
      for site in list(pending):
          s = slab_rel.copy()
          s.info["adsorbate_info"] = slab.info["adsorbate_info"]
          add_adsorbate(s, "Hg", height=0.0, position=site)
          s.positions[-1, 2] = top_z(slab_rel) + h_start(site)
          i = len(s) - 1
          s.set_constraint([FixAtoms(mask=np.append(tags >= 3, False)),
                            FixCartesian(i, mask=(True, True, False))])
          r, flag = relax(s, f"site_{site}", func, on_busy=on_busy)
          if r is None:
              continue
          pending.remove(site)
          at = r["atoms"]
          au = [a.index for a in at if a.symbol == "Au"]
          h = at.positions[i, 2] - top_z(at)
          d = at.get_distances(i, au, mic=True).min()
          refs = {"A_isolated_atom": e_slab + e_atom, "B_far_above_slab": e_far,
                  "C_bulkHg_plus_Ecoh_exp": e_slab + e_hgb + 0.67}
          for scheme, eref in refs.items():
              rows.append(dict(method=func, a0_A=a0, site=site, scheme=scheme,
                               E_ads_eV=r["energy_eV"] - eref, E_total_eV=r["energy_eV"],
                               E_ref_eV=eref, height_A=h, d_AuHg_min_A=d, status=flag,
                               n_ionic=r["n_scf"]))
          print(f"  {site}: E_ads(A)={r['energy_eV']-refs['A_isolated_atom']:+.4f} "
                f"E_ads(B)={r['energy_eV']-e_far:+.4f} h={h:.3f} d={d:.3f}", flush=True)
    return rows, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--functional", choices=["PBE", "PBE-D3"])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--sites", nargs="+", default=None,
                    help="site order for this worker (default ontop bridge fcc hcp)")
    a = ap.parse_args()
    if a.plan:
        print("per functional: 9 bulk-Au SCF (1 atom), slab relax, Hg atom SCF, alpha-Hg vc-relax, "
              "far relax, 4 site relaxes (37 atoms)")
        print(PROD)
    else:
        PROD["engine"] = a.engine
        if a.sites:
            PROD["sites"] = a.sites
        rows, info = run(a.functional)
        out = os.path.join(RES, "dft_results.csv")
        old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        if len(old):
            new_keys = {(a.functional, r["site"]) for r in rows}
            old = old[[(m, st) not in new_keys for m, st in zip(old.method, old.site)]]
        pd.concat([old, pd.DataFrame(rows)]).to_csv(out, index=False, float_format="%.6f")
        outb = os.path.join(RES, "dft_bulk_refs.csv")
        oldb = pd.read_csv(outb) if os.path.exists(outb) else pd.DataFrame()
        if len(oldb):
            oldb = oldb[oldb.functional != a.functional]
        pd.concat([oldb, pd.DataFrame([info])]).to_csv(outb, index=False, float_format="%.6f")
