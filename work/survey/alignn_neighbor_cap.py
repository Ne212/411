"""ALIGNN-FF max_neighbors=52 vs the number of neighbours inside the 5 A cutoff.
Bulk fcc metal at the model's own a0, 2x2x2 conventional supercell, 20 random rattles of 1e-4 A:
a smooth PES changes by << 1 meV; a neighbour-cap discontinuity gives jumps of eV.
Usage: <alignn venv>/bin/python alignn_neighbor_cap.py MODEL_KEY"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcs import make_calc
key = sys.argv[1]; calc, _ = make_calc(key)
a0s = json.load(open(f"../../results/published/models/{key}/settings.json"))["a0"]
for el, a0 in sorted(a0s.items()):
    s = bulk(el, "fcc", a=a0, cubic=True).repeat(2)
    nn = np.bincount(neighbor_list("i", s, 5.0)).max()
    E = []
    for k in range(20):
        a = s.copy(); a.rattle(1e-4, seed=k); a.calc = calc; E.append(a.get_potential_energy())
    print(f"{key:22s} {el}: a0={a0:.3f} A, 4th shell a0*sqrt2={a0*np.sqrt(2):.3f} A, neighbours within 5 A = {nn} "
          f"(cap 52), spread of E over 20 rattles = {np.ptp(E):.4f} eV")
