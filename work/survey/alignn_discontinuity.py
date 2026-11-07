"""Energy of an ALIGNN-FF slab vs a tiny displacement of one surface atom (tests for PES discontinuities).
Usage: <alignn venv>/bin/python alignn_discontinuity.py MODEL_KEY METAL FACET(111|001) NX"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
from ase.build import fcc100, fcc111
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcs import make_calc
key, el, fac, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a0 = json.load(open(f"../../results/published/models/{key}/settings.json"))["a0"][el]
s = (fcc100 if fac == "001" else fcc111)(el, (n, n, 4), a=a0, vacuum=12.0)
calc, _ = make_calc(key); i = int(np.where(s.get_tags() == 1)[0][0])
E = []
for dz in np.linspace(-0.01, 0.01, 41):
    a = s.copy(); a.positions[i, 2] += dz; a.calc = calc; E.append(a.get_potential_energy())
E = np.array(E); jumps = np.abs(np.diff(E))
print(f"{key} {el}({fac}) {n}x{n}: E range over dz=±0.01 A: {E.max()-E.min():.3f} eV; max |dE| between "
      f"neighbouring points (0.0005 A apart): {jumps.max():.3f} eV; smooth expectation ~ fmax*0.0005 = {0.3*0.0005:.5f} eV")
