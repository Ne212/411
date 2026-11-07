"""Reproduce the first energy jump of the ALIGNN-FF-mptrj clean Ni(001) c(4x4) relaxation and inspect it."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.build import fcc100
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.neighborlist import neighbor_list
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcs import make_calc
key = "ALIGNN-FF-mptrj"; calc, _ = make_calc(key)
a0 = json.load(open(f"../../results/published/models/{key}/settings.json"))["a0"]["Ni"]
s = fcc100("Ni", (4, 4, 4), a=a0, vacuum=12.0); s.set_constraint(FixAtoms(mask=s.get_tags() >= 3)); s.calc = calc
frames = []
opt = BFGS(s, logfile=None)
for k, _ in zip(range(4), opt.irun(fmax=0.02, steps=3)):
    frames.append((s.positions.copy(), s.get_potential_energy(), np.abs(s.get_forces()).max()))
for k in range(1, len(frames)):
    (p0, e0, f0), (p1, e1, f1) = frames[k - 1], frames[k]
    d = np.linalg.norm(p1 - p0, axis=1)
    a, b = s.copy(), s.copy(); a.positions, b.positions = p0, p1
    n0 = np.bincount(neighbor_list("i", a, 5.0), minlength=len(a)); n1 = np.bincount(neighbor_list("i", b, 5.0), minlength=len(b))
    print(f"step {k}: dE = {e1-e0:+.3f} eV, max atom move = {d.max():.4f} A, neighbours(5 A) changed for "
          f"{int((n0 != n1).sum())} atoms, max neighbours {n0.max()}->{n1.max()}")
# energy along the straight line between the two frames of the first jump
(p0, e0, _), (p1, e1, _) = frames[1], frames[2]
for t in np.linspace(0, 1, 11):
    a = s.copy(); a.set_constraint(); a.positions = p0 + t * (p1 - p0); a.calc = calc
    print(f"  t={t:.1f} E={a.get_potential_energy():.3f}")
# pairs inside 5.0 A at t=0.7 and t=0.8, and the distances that cross the cutoff
for t in (0.7, 0.8):
    a = s.copy(); a.set_constraint(); a.positions = p0 + t * (p1 - p0)
    i, j, dd = neighbor_list("ijd", a, 5.0)
    print(f"t={t}: pairs within 5.0 A = {len(i)}, largest pair distance inside = {dd.max():.4f} A")
a = s.copy(); a.set_constraint(); a.positions = p0 + 0.8 * (p1 - p0)
i, j, dd = neighbor_list("ijd", a, 5.0); new = dd[dd > 4.99]
print("distances in (4.99, 5.0] at t=0.8:", np.round(np.sort(new)[:8], 4), "count", len(new))
for rc in (3.0, 4.0, 5.0):
    c = []
    for t in (0.7, 0.8):
        a = s.copy(); a.set_constraint(); a.positions = p0 + t * (p1 - p0)
        c.append(len(neighbor_list("i", a, rc)))
    print(f"pairs within {rc} A: t=0.7 {c[0]}, t=0.8 {c[1]}")
a = s.copy(); a.set_constraint(); a.positions = p0 + 0.75 * (p1 - p0)
i, j, dd = neighbor_list("ijd", a, 4.2); print("distances near 3.0 / 4.0 A at t=0.75:", np.round(np.sort(dd[(np.abs(dd-3.0)<0.05)|(np.abs(dd-4.0)<0.05)])[:6], 4))
