"""Which heavy / superheavy elements does a model ACCEPT (evaluates an Au-X dimer without error)?
Only "accepted"/"rejected" is recorded. Energies are deliberately not stored: no model has superheavy
elements (Z >= 104) in its training data, so any number it returns is not a prediction.
Usage: <venv>/bin/python element_coverage.py MODEL_KEY  -> one JSON line"""
import json, sys, os, warnings
warnings.filterwarnings("ignore")
from ase import Atoms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcs import make_calc
EL = ["Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
key = sys.argv[1]; calc, _ = make_calc(key); out = {}
for el in EL:
    try:
        a = Atoms(["Au", el], positions=[(0, 0, 0), (0, 0, 2.8)], cell=[20] * 3, pbc=True); a.center(); a.calc = calc
        a.get_potential_energy(); out[el] = "accepted"
    except Exception as exc:
        out[el] = f"rejected ({type(exc).__name__})"
print(json.dumps({"model": key, **out}))
