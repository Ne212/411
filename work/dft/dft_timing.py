#!/usr/bin/env python
"""
Timing test: one SCF of Hg@fcc on Au(111) 3x3x4 (vacuum 12 A each side, a = 4.16 A guess,
Hg 2.4 A above the top layer), PBE, 65/390 Ry, k = 4x4x1, on different engines.
Usage: python dft_timing.py cpu 2      (engine, npool)
       python dft_timing.py gpu 1
"""
import sys

from ase.build import add_adsorbate, fcc111
from qe_common import run_pw

engine, npool = sys.argv[1], int(sys.argv[2])
func = sys.argv[3] if len(sys.argv) > 3 else "PBE"
s = fcc111("Au", size=(3, 3, 4), a=4.16, vacuum=12.0)
add_adsorbate(s, "Hg", height=2.4, position="fcc")
r = run_pw(s, f"timing_{func}_{engine}_nk{npool}", func, kpts=(4, 4, 1), engine=engine, npool=npool)
print({k: v for k, v in r.items() if k != "atoms"})
