"""
Decompose the difference between Miro's ALIGNN-FF script and the 411 protocol by changing ONE
setting at a time (same calculator throughout).  Sites are placed with ASE add_adsorbate names, so
'fcc'/'hcp' are the true hollow sites of the ASE fcc111 slab.
  S0  Miro's settings: 3x3x3, a=4.08, vacuum 10, bottom 2 layers fixed, clean slab NOT relaxed,
      Hg free in x,y,z, fmax 0.05 (<=200 steps), Hg atom in a 10 A box
  S1  S0 + clean slab relaxed (same constraints)
  S2  S1 + Au lattice constant = the model's own a0
  S3  S2 + 3x3x4 slab, vacuum 12 (bottom 2 of 4 fixed)
  S4  S3 + fmax 0.02 (<=1000 steps), Hg laterally fixed, Hg atom in 15 A box  (= 411 protocol)
Usage: <venv>/bin/python protocol_bridge.py MODEL_KEY  -> results/hg_au/<key>/protocol_bridge.csv
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms, FixedLine
from ase.optimize import BFGS
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from calcs import make_calc

key = sys.argv[1]
out = os.path.join(HERE, "..", "..", "results", "hg_au", key)
a0 = float(pd.read_csv(os.path.join(out, "bulk.csv"))["a0_A"].iloc[0])     # from hg_au_protocol.py
calc, info = make_calc(key)

STEPS = [
    dict(step="S0_miro", layers=3, a=4.08, vac=10.0, relax_clean=False, fmax=0.05, nmax=200, zonly=False, box=10.0),
]
STEPS.append(dict(STEPS[-1], step="S1_clean_slab_relaxed", relax_clean=True))
STEPS.append(dict(STEPS[-1], step="S2_own_a0", a=a0))
STEPS.append(dict(STEPS[-1], step="S3_4_layers_vac12", layers=4, vac=12.0))
STEPS.append(dict(STEPS[-1], step="S4_fmax0.02_zonly_box15(=411)", fmax=0.02, nmax=1000, zonly=True, box=15.0))

rows = []
for st in STEPS:
    slab = fcc111("Au", size=(3, 3, st["layers"]), a=st["a"], vacuum=st["vac"])
    fixed = slab.get_tags() >= st["layers"] - 1          # bottom two layers
    slab.set_constraint(FixAtoms(mask=fixed)); slab.calc = calc
    if st["relax_clean"]:
        BFGS(slab, logfile=None).run(fmax=st["fmax"], steps=st["nmax"])
    e_clean = slab.get_potential_energy()
    hg = Atoms("Hg", cell=[st["box"]] * 3, pbc=True)
    if st["box"] == 15.0:
        hg.center()                                       # as in the 411 protocol; Miro: Hg at origin
    hg.calc = calc; e_hg = hg.get_potential_energy()
    for site in ("ontop", "bridge", "fcc", "hcp"):
        s = slab.copy(); add_adsorbate(s, "Hg", height=2.5, position=site)
        i = len(s) - 1; xy0 = s.positions[i, :2].copy()
        cons = [FixAtoms(mask=np.r_[fixed, False])]
        if st["zonly"]:
            cons.append(FixedLine(i, [0, 0, 1]))
        s.set_constraint(cons); s.calc = calc
        opt = BFGS(s, logfile=None); conv = bool(opt.run(fmax=st["fmax"], steps=st["nmax"]))
        e = s.get_potential_energy()
        rows.append(dict(model=key, **st, site=site, E_ads_eV=e - e_clean - e_hg, converged=conv,
                         n_steps=opt.nsteps, lateral_drift_A=float(np.linalg.norm(s.positions[i, :2] - xy0))))
    print(st["step"], {r["site"]: round(r["E_ads_eV"], 3) for r in rows[-4:]}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "protocol_bridge.csv"), index=False, float_format="%.5f")
print(df.pivot(index="step", columns="site", values="E_ads_eV").round(3).loc[[s["step"] for s in STEPS]])
