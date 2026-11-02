#!/usr/bin/env python
"""
Sensitivity check linking the consistent benchmark to the upstream numbers.
Same protocol as mlip_benchmark.py (scheme A, Hg laterally fixed, bottom 2 layers fixed),
varying only (i) the Au lattice constant (ASE default 4.08 A vs. the method's own a0) and
(ii) the force threshold (0.05 eV/A as upstream CHGNet vs. 0.02 eV/A).
Output: ../results/mlip_sensitivity.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms, FixedLine
from ase.optimize import BFGS

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
bulk = pd.read_csv(os.path.join(RES, "mlip_bulk.csv")).set_index("method")


def calcs():
    from chgnet.model.dynamics import CHGNetCalculator
    from mace.calculators import mace_mp
    yield "CHGNet", CHGNetCalculator(use_device="cuda", check_cuda_mem=False)
    yield "MACE", mace_mp(model="medium", device="cuda", default_dtype="float64")


def relax(at, calc, fmax):
    at.calc = calc
    BFGS(at, logfile=None).run(fmax=fmax, steps=1000)
    return at.get_potential_energy()


rows = []
for name, calc in calcs():
    hg = Atoms("Hg", pbc=True, cell=[15] * 3); hg.center(); hg.calc = calc
    e_atom = hg.get_potential_energy()
    for a_label, a in [("ASE default 4.08", 4.08), ("own a0", float(bulk.loc[name, "a0_A"]))]:
        for fmax in (0.05, 0.02):
            slab = fcc111("Au", size=(3, 3, 4), a=a, vacuum=12.0)
            slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))
            e_slab = relax(slab, calc, fmax)
            for site in ("ontop", "bridge", "fcc", "hcp"):
                s = slab.copy()
                add_adsorbate(s, "Hg", 2.5, site)
                s.set_constraint([FixAtoms(mask=s.get_tags() >= 3), FixedLine(len(s) - 1, [0, 0, 1])])
                e = relax(s, calc, fmax)
                rows.append(dict(method=name, lattice=a_label, a_A=a, fmax=fmax, site=site,
                                 E_ads_A_eV=e - e_slab - e_atom))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "mlip_sensitivity.csv"), index=False, float_format="%.6f")
