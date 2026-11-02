#!/usr/bin/env python
"""
Convergence tests for the DFT reference (PBE unless stated), on cheap models.

1. Au bulk (1-atom fcc): a0 from a Birch-Murnaghan fit at ecutwfc 50/65/80 Ry and k 16^3/20^3,
   for PBE and PBE-D3.
2. Test model Au(111) 2x2x3, a = PBE a0, vacuum 8 A each side, rigid (unrelaxed) geometry,
   Hg at fcc (2.4 A) and ontop (2.6 A). Quantities: E_ads(fcc), E_ads(ontop) vs Hg atom in a
   15 A box (Gamma), and dE = E(fcc) - E(ontop).
     a) cutoff: (50,300) (65,390) (80,480) Ry, k 6x6x1  [dual 6 for the Hg USPP]
     b) ecutrho at fixed ecutwfc 65: 390 vs 520
     c) k-mesh at 65/390: 4x4, 6x6, 9x9 (2x2 cell; ~ 2.7x2.7, 4x4, 6x6 on the 3x3 cell)
     d) vacuum 8 vs 12 A each side; dipole correction on/off
     e) PBE-D3(BJ) at 65/390, k 6x6 (dispersion contribution)
Output: ../../results/dft_convergence.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111
from ase.eos import EquationOfState

from qe_common import run_pw

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
rows = []


def cut(ecw, ecr):
    return {"system": {"ecutwfc": ecw, "ecutrho": ecr}}


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out.setdefault(k, {}).update(v)
    return out


# ---------------------------------------------------------------- 1. bulk Au
def au_a0(func, ecw, ecr, k):
    a_grid = np.linspace(4.00, 4.30, 9) if func == "PBE" else np.linspace(3.95, 4.25, 9)
    vols, ens = [], []
    for a in a_grid:
        r = run_pw(bulk("Au", "fcc", a=a), f"conv/bulkAu_{func}_{ecw:.0f}_{ecr:.0f}_k{k}_a{a:.4f}",
                   func, kpts=(k, k, k), overrides=cut(ecw, ecr), npool=5)
        vols.append(r["atoms"].get_volume()); ens.append(r["energy_eV"])
    v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
    return (4 * v0) ** (1 / 3), B * 160.21766208


a0_pbe = None
for func in ("PBE", "PBE-D3"):
    for ecw, ecr, k in [(50, 300, 16), (65, 390, 16), (80, 480, 16), (65, 390, 20)]:
        a0, B = au_a0(func, ecw, ecr, k)
        rows.append(dict(test="bulk_Au", functional=func, ecutwfc=ecw, ecutrho=ecr, kpts=f"{k}^3",
                         quantity="a0_A", value=a0))
        rows.append(dict(test="bulk_Au", functional=func, ecutwfc=ecw, ecutrho=ecr, kpts=f"{k}^3",
                         quantity="B0_GPa", value=B))
        print(func, ecw, ecr, k, f"a0={a0:.4f} B0={B:.1f}", flush=True)
        if func == "PBE" and (ecw, k) == (65, 16):
            a0_pbe = a0
pd.DataFrame(rows).to_csv(os.path.join(RES, "dft_convergence.csv"), index=False, float_format="%.6f")
if "--bulk-only" in sys.argv:
    sys.exit()


# ---------------------------------------------------------------- 2. test model
def model(site, vac):
    s = fcc111("Au", size=(2, 2, 3), a=a0_pbe, vacuum=vac)
    if site:
        add_adsorbate(s, "Hg", height=2.4 if site == "fcc" else 2.6, position=site)
    return s


def eads_set(tag, func, ecw, ecr, k, vac=8.0, extra=None):
    ov = merge(cut(ecw, ecr), extra or {})
    base = f"conv/{tag}_{func}_{ecw:.0f}_{ecr:.0f}_k{k}_v{vac:.0f}"
    e = {}
    for site in (None, "fcc", "ontop"):
        r = run_pw(model(site, vac), f"{base}_{site or 'slab'}", func, kpts=(k, k, 1), overrides=ov,
                   npool=2)
        e[site] = r["energy_eV"]
        print(f"  {base}_{site or 'slab'}: E={r['energy_eV']:.6f} iters={r['scf_iterations']} "
              f"{r['wall_pw']}", flush=True)
    hg = Atoms("Hg", pbc=True, cell=[15.0] * 3); hg.center()
    ra = run_pw(hg, f"conv/atom_{func}_{ecw:.0f}_{ecr:.0f}", func, kpts=None,
                overrides=cut(ecw, ecr))
    out = dict(E_ads_fcc=e["fcc"] - e[None] - ra["energy_eV"],
               E_ads_ontop=e["ontop"] - e[None] - ra["energy_eV"],
               dE_fcc_minus_ontop=e["fcc"] - e["ontop"])
    for q, v in out.items():
        rows.append(dict(test=tag, functional=func, ecutwfc=ecw, ecutrho=ecr, kpts=f"{k}x{k}x1",
                         vacuum=vac, quantity=q, value=v))
    print(tag, func, ecw, ecr, k, vac, {q: round(v, 4) for q, v in out.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "dft_convergence.csv"), index=False,
                              float_format="%.6f")


for ecw, ecr in [(50, 300), (65, 390), (80, 480)]:
    eads_set("cutoff", "PBE", ecw, ecr, 6)
eads_set("ecutrho", "PBE", 65, 520, 6)
for k in (4, 9):
    eads_set("kmesh", "PBE", 65, 390, k)
eads_set("vacuum", "PBE", 65, 390, 6, vac=12.0)
dip = {"control": {"tefield": True, "dipfield": True},
       "system": {"edir": 3, "eamp": 0.0, "emaxpos": 0.95, "eopreg": 0.04}}
eads_set("dipole", "PBE", 65, 390, 6, extra=dip)
eads_set("D3", "PBE-D3", 65, 390, 6)
print("done")
