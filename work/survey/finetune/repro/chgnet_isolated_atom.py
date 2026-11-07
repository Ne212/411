"""Minimal reproduction: CHGNet's StructureData silently replaces a structure that contains an isolated atom
(no neighbour within the 6 A atom-graph cutoff) by a random other structure.  chgnet 0.4.2."""
import warnings; warnings.filterwarnings("ignore")
import random
from pymatgen.core import Lattice, Structure
from chgnet.data.dataset import StructureData

au = Structure(Lattice.cubic(4.08), ["Au"] * 4, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
hg = Structure(Lattice.cubic(15.0), ["Hg"], [[0.5, 0.5, 0.5]])          # free Hg atom in a 15 A box
ds = StructureData(structures=[au, hg], energies=[-3.2, 0.3], forces=[[[0, 0, 0]] * 4, [[0, 0, 0]]], shuffle=False)
random.seed(0)
for idx in (0, 1):
    graph, targets = ds[idx]
    print(f"ds[{idx}] requested {[au, hg][idx].composition.formula:>4s} -> returned graph with "
          f"{graph.composition} (Z={sorted(set(graph.atomic_number.tolist()))}), "
          f"target e = {float(targets['e']):+.2f}")
print("failed_graph_id:", ds.failed_graph_id, "| failed_idx:", ds.failed_idx)

# second case: Hg 10 A above an Au(111) slab (the 'far' frame) - the Hg atom is isolated inside the slab cell
from ase.build import fcc111, add_adsorbate
from pymatgen.io.ase import AseAtomsAdaptor
slab = fcc111("Au", (3, 3, 4), a=4.15, vacuum=12.0); add_adsorbate(slab, "Hg", 10.0, "fcc"); slab.pbc = True
far = AseAtomsAdaptor.get_structure(slab)
ds2 = StructureData(structures=[au, far], energies=[-3.2, -3.0], forces=[[[0, 0, 0]] * 4, [[0, 0, 0]] * len(far)],
                    shuffle=False)
graph, targets = ds2[1]
print(f"ds2[1] requested {far.composition.formula} -> returned graph with {graph.composition}, "
      f"target e = {float(targets['e']):+.2f}; failed_graph_id: {ds2.failed_graph_id}")
