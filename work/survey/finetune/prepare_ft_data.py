"""
Stage 4 data: QE-PBE outputs of project 411 (work/dft/runs/prod/PBE, one consistent setting: 50/300 Ry)
-> extxyz for fine-tuning one foundation model.

Split by calculation type (no frame of a held-out calculation is ever trained on):
  test       : every frame of the held-out site trajectory (default: site_ontop, 16 frames)
  atom group : Hg_atom (isolated Hg, 15 A box) + far (Hg 10 A above the slab). 'far' carries the same information
               as the free atom for a local model, so both are excluded from the "noatom" variant.
  pool       : slab, the other three sites, alpha-Hg vc-relax, bulk-Au E(V)  -> 90 % train / 10 % valid
               (random, seed 0, same split for every model)
Energy zero: QE total energies are shifted per element, E_target = E_QE + n_Au*d_Au + n_Hg*d_Hg, with d fitted by
least squares to the foundation model's predictions on the condensed-phase POOL frames only (not test, not atom
group). The same shift is used for both variants, so the free-atom information enters only through the atom
group frames.
Usage: <venv>/bin/python prepare_ft_data.py MODEL_KEY [--test-site site_ontop]
Output: work/survey/finetune/data/<MODEL_KEY>/{train_noatom,train_atom,valid,test,atom_group}.xyz, prep.json
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from calcs import make_calc
QE = os.path.join(HERE, "..", "..", "dft", "runs", "prod", "PBE")
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--test-site", default="site_ontop")
args = ap.parse_args()

calcs = sorted(d for d in os.listdir(QE) if os.path.isfile(os.path.join(QE, d, "pw.out")))
frames = []
for d in calcs:
    for k, a in enumerate(read(os.path.join(QE, d, "pw.out"), index=":", format="espresso-out")):
        a.info.update(calc=d, step=k, E_QE=float(a.get_potential_energy())); a.arrays["F_QE"] = a.get_forces().copy()
        a.calc = None; frames.append(a)
group = lambda a: ("test" if a.info["calc"] == args.test_site else
                   "atom" if a.info["calc"] in ("Hg_atom", "far") else "pool")
rng = np.random.default_rng(0)
pool = [a for a in frames if group(a) == "pool"]
val_idx = set(rng.choice(len(pool), size=max(1, round(0.1 * len(pool))), replace=False).tolist())

calc, info = make_calc(args.model)
pred = []
for a in pool:
    b = a.copy(); b.calc = calc; pred.append(b.get_potential_energy())
X = np.array([[np.sum(a.numbers == 79), np.sum(a.numbers == 80)] for a in pool], float)
y = np.array(pred) - np.array([a.info["E_QE"] for a in pool])
d, *_ = np.linalg.lstsq(X, y, rcond=None)
shift = lambda a: np.sum(a.numbers == 79) * d[0] + np.sum(a.numbers == 80) * d[1]

out = os.path.join(HERE, "data", args.model); os.makedirs(out, exist_ok=True)
sets = {"train": [], "valid": [], "test": [], "atom_group": []}
for a in frames:
    b = a.copy(); b.info = dict(a.info); b.info["REF_energy"] = a.info["E_QE"] + shift(a)
    b.arrays["REF_forces"] = a.arrays["F_QE"]; b.info["config_type"] = a.info["calc"]
    g = group(a)
    if g == "pool":
        sets["valid" if pool.index(a) in val_idx else "train"].append(b)
    else:
        sets["test" if g == "test" else "atom_group"].append(b)
write(os.path.join(out, "train_noatom.xyz"), sets["train"], format="extxyz")
write(os.path.join(out, "train_atom.xyz"), sets["train"] + sets["atom_group"], format="extxyz")
for k in ("valid", "test", "atom_group"):
    write(os.path.join(out, f"{k}.xyz"), sets[k], format="extxyz")
meta = dict(model=args.model, test_site=args.test_site, shift_eV_per_atom=dict(Au=d[0], Hg=d[1]),
            shift_fit_rms_eV=float(np.sqrt(np.mean((y - X @ d) ** 2))),
            n=dict(train=len(sets["train"]), valid=len(sets["valid"]), test=len(sets["test"]),
                   atom_group=len(sets["atom_group"])),
            train_calcs=sorted({a.info["calc"] for a in sets["train"]}), qe_dir=os.path.abspath(QE))
json.dump(meta, open(os.path.join(out, "prep.json"), "w"), indent=1, default=float)
print(json.dumps(meta, indent=1, default=float))
