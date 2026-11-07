"""Classify Miro's ALIGNN-FF site coordinates against the layers of the 3x3x3 ASE fcc111 slab."""
import numpy as np
from ase.build import fcc111
from ase.geometry import find_mic

slab = fcc111('Au', size=(3, 3, 3), a=4.08, vacuum=10.0)
tags = slab.get_tags()                      # 1 = top, 2 = 2nd, 3 = 3rd layer
sites = {"ontop": (7.212, 2.498), "bridge": (6.491, 1.249), "fcc": (5.770, 1.666), "hcp": (5.289, 0.833),
         "hcp_final": (5.66, 1.65)}
for name, xy in sites.items():
    out = []
    for t in (1, 2, 3):
        d = slab.positions[tags == t, :2] - np.array(xy)
        d3 = np.c_[d, np.zeros(len(d))]
        dm, l = find_mic(d3, slab.cell, pbc=[True, True, False])
        out.append(f"layer{t} min lateral dist {l.min():.3f}")
    print(f"{name:9s}", " | ".join(out))
# fixed atoms in Miro's mask
z = slab.positions[:, 2]; zs = np.sort(z); lt = (zs[-1] - zs[0]) / 3.0
mask = z < zs[0] + 2 * lt + 0.1
print("fixed per layer tag:", {t: int(mask[tags == t].sum()) for t in (1, 2, 3)})
