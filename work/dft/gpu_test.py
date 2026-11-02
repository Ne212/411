"""GPU speed test: 4 SCF iterations of the PBE hcp starting structure, production settings."""
import pandas as pd
from ase.build import add_adsorbate, fcc111
from ase.io import read
from qe_common import run_pw
slab = read("runs/prod/PBE/slab/pw.out", index=-1, format="espresso-out")
ref = fcc111("Au", size=(3, 3, 4), a=4.1503, vacuum=12.0)
slab.info["adsorbate_info"] = ref.info["adsorbate_info"]
d = pd.read_csv("../../results/mlip_results.csv")
h = d[(d.method == "MACE") & (d.site == "hcp") & (d.scheme == "A_isolated_atom")].height_A.iloc[0]
add_adsorbate(slab, "Hg", height=0.0, position="hcp")
slab.positions[-1, 2] = slab.positions[:36, 2].max() + h
r = run_pw(slab, "gpu_test/PBE_hcp_4iter", "PBE", kpts=(4, 4, 1), engine="gpu",
           overrides={"system": {"ecutwfc": 50.0, "ecutrho": 300.0}, "electrons": {"electron_maxstep": 4}})
print({k: v for k, v in r.items() if k != "atoms"})
