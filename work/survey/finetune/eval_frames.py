"""
Frame-level errors before/after fine-tuning on the 411 QE frames, per split
(train, valid, test = held-out site, atom_group = isolated Hg + far).
Targets are the per-element shifted QE energies of the base model (prepare_ft_data.py), so the foundation
model's errors are already minimised by the best constant per-element shift.
Usage: <venv>/bin/python eval_frames.py MODEL_KEY BASE_DATA_DIR_NAME
       e.g. eval_frames.py MACE-MP-0-ft-atom MACE-MP-0-medium
Output: results/finetune/frames_<MODEL_KEY>.csv (one row per frame)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ase.io import read
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from calcs import make_calc
key, base = sys.argv[1], sys.argv[2]
D = os.path.join(HERE, "data", base); OUT = os.path.join(HERE, "..", "..", "..", "results", "finetune")
os.makedirs(OUT, exist_ok=True)
calc, _ = make_calc(key)
rows = []
for split, f in (("train", "train_noatom.xyz"), ("valid", "valid.xyz"), ("test", "test.xyz"), ("atom_group", "atom_group.xyz")):
    for a in read(os.path.join(D, f), index=":"):
        ref_e, ref_f = a.info["REF_energy"], a.arrays["REF_forces"]
        b = a.copy(); b.calc = calc
        e = b.get_potential_energy(); fo = b.get_forces()
        rows.append(dict(model=key, split=split, calc=a.info["calc"], step=a.info["step"], natoms=len(a),
                         dE_per_atom=(e - ref_e) / len(a), dE_total=e - ref_e,
                         F_MAE=float(np.abs(fo - ref_f).mean()), F_max_err=float(np.abs(fo - ref_f).max())))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, f"frames_{key}.csv"), index=False, float_format="%.6f")
s = df.groupby("split").agg(E_MAE_meV_atom=("dE_per_atom", lambda x: 1000 * np.abs(x).mean()),
                            F_MAE_meV_A=("F_MAE", lambda x: 1000 * x.mean()), n=("natoms", "size"))
print(key); print(s.round(2).to_string())
