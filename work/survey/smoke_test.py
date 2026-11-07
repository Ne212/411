"""Smoke test of one model: builds the calculator from calcs.py and evaluates a few tiny systems.
Usage: <venv>/bin/python smoke_test.py MODEL_KEY      -> results/survey_logs/smoke/<key>.json"""
import json, os, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, add_adsorbate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcs import make_calc

key = sys.argv[1]
out = os.path.join(os.path.dirname(__file__), "..", "..", "results", "survey_logs", "smoke")
os.makedirs(out, exist_ok=True)
res = dict(model=key, ok=False)
t0 = time.time()
try:
    calc, info = make_calc(key)
    res["info"] = info
    res["t_load_s"] = round(time.time() - t0, 1)
    au = bulk("Au", "fcc", a=4.08); au.calc = calc
    res["E_Au_bulk_a4.08_per_atom"] = float(au.get_potential_energy() / len(au))
    for el in ("Hg", "Au"):
        a = Atoms(el, cell=[15.0] * 3, pbc=True); a.center(); a.calc = calc
        res[f"E_{el}_atom_15A"] = float(a.get_potential_energy())
    sup = bulk("Au", "fcc", a=4.08, cubic=True).repeat(2); sup.rattle(0.05, seed=1); sup.calc = calc
    f = sup.get_forces(); res["sum_forces_rattled_Au32"] = float(np.abs(f.sum(0)).max())
    # stress check: analytic vs central finite difference (catches unit bugs, e.g. GPa vs eV/A^3)
    st = bulk("Au", "fcc", a=4.12); st.set_cell(st.cell @ [[1, 0.02, 0], [0, 1, 0], [0, 0, 1]], scale_atoms=True)
    st.calc = calc
    sa = st.get_stress(); sn = calc.calculate_numerical_stress(st, d=1e-4)
    res["stress_analytic_eVA3"] = sa.tolist(); res["stress_numeric_eVA3"] = sn.tolist()
    res["stress_max_abs_diff_eVA3"] = float(np.abs(sa - sn).max())
    s = fcc111("Au", size=(3, 3, 4), a=4.08, vacuum=12.0); add_adsorbate(s, "Hg", 2.5, "fcc"); s.calc = calc
    s.get_potential_energy()           # warm-up (compilation, graph build)
    s.rattle(1e-3, seed=2); t1 = time.time(); e = s.get_potential_energy(); s.get_forces()
    res["t_single_point_37atoms_s"] = round(time.time() - t1, 4)
    res["E_slab_Hg_37"] = float(e)
    res["ok"] = True
except Exception as exc:
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()[-3000:]
try:
    import torch
    res["torch"] = torch.__version__; res["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        res["gpu_mem_peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
except ImportError:
    pass
res["t_total_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(out, f"{key}.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, default=str))
