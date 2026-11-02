#!/usr/bin/env python
"""
Build LaTeX tables (report/tables/*.tex) and the combined comparison CSV
(results/comparison_table.csv) from results/*.csv. Nothing is typed by hand.

Scheme D (diagnostic, needs DFT): MLIP adsorption energy referenced to the MLIP's own bulk alpha-Hg
plus the PBE cohesive energy of Hg:  E_ads^D = E(slab+Hg) - E(slab) - E_bulk^MLIP(Hg) - E_coh^PBE(Hg).
It replaces the MLIP's free-atom energy (never trained) by a PBE one, transferred through bulk Hg.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "report", "tables")
os.makedirs(TAB, exist_ok=True)
SITES = ["ontop", "bridge", "fcc", "hcp"]
NAME = {"CHGNet": "CHGNet", "MACE": "MACE-MP-0", "PBE": "PBE", "PBE-D3": "PBE-D3(BJ)"}
SCH = {"A_isolated_atom": "A", "B_far_above_slab": "B", "C_bulkHg_plus_Ecoh_exp": "C",
       "D_bulkHg_plus_Ecoh_PBE": "D"}

mlip = pd.read_csv(os.path.join(RES, "mlip_results.csv"))
mlip = mlip[mlip.method != "LJ"]
mb = pd.read_csv(os.path.join(RES, "mlip_bulk.csv")).set_index("method")
dft_p = os.path.join(RES, "dft_results.csv")
dft = pd.read_csv(dft_p) if os.path.exists(dft_p) else pd.DataFrame()
db_p = os.path.join(RES, "dft_bulk_refs.csv")
db = pd.read_csv(db_p).set_index("functional") if os.path.exists(db_p) else pd.DataFrame()

allr = pd.concat([mlip, dft], ignore_index=True)

# scheme D for MLIPs
if "PBE" in db.index:
    ecoh_pbe = db.loc["PBE", "Ecoh_Hg"]
    extra = []
    for m in ["CHGNet", "MACE"]:
        a = mlip[(mlip.method == m) & (mlip.scheme == "A_isolated_atom")]
        for _, r in a.iterrows():
            eref = mb.loc[m, "E_slab"] + mb.loc[m, "E_Hg_bulk_per_atom"] + ecoh_pbe
            rr = r.copy(); rr["scheme"] = "D_bulkHg_plus_Ecoh_PBE"
            rr["E_ref_eV"] = eref; rr["E_ads_eV"] = r["E_total_eV"] - eref
            extra.append(rr)
    allr = pd.concat([allr, pd.DataFrame(extra)], ignore_index=True)

cols = ["method", "site", "scheme", "E_ads_eV", "height_A", "d_AuHg_min_A", "a0_A"]
allr[cols].sort_values(["method", "site", "scheme"]).to_csv(
    os.path.join(RES, "comparison_table.csv"), index=False, float_format="%.4f")

methods = [m for m in ["CHGNet", "MACE", "PBE", "PBE-D3"] if m in set(allr.method)]


def f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---- main comparison table
lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Method & Site & \multicolumn{4}{c}{$E_\mathrm{ads}$ (eV), scheme} & $h$ (\AA) & $d_\mathrm{Au-Hg}$ (\AA)\\",
         r"\cmidrule(lr){3-6}", r" & & A & B & C & D & & \\", r"\midrule"]
for m in methods:
    for k, s in enumerate(SITES):
        sub = allr[(allr.method == m) & (allr.site == s)].set_index("scheme")
        if sub.empty:
            continue
        g = lambda sc: sub.loc[sc, "E_ads_eV"] if sc in sub.index else np.nan
        A = sub.loc["A_isolated_atom"]
        lines.append(f"{NAME[m] if k == 0 else ''} & {s} & {f(g('A_isolated_atom'))} & "
                     f"{f(g('B_far_above_slab'))} & {f(g('C_bulkHg_plus_Ecoh_exp'))} & "
                     f"{f(g('D_bulkHg_plus_Ecoh_PBE'))} & {f(A['height_A'], 2)} & {f(A['d_AuHg_min_A'], 2)}\\\\")
    lines.append(r"\midrule" if m != methods[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
open(os.path.join(TAB, "tab_main.tex"), "w").write("\n".join(lines) + "\n")

# ---- bulk / reference table
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Method & $a_0$(Au) (\AA) & $B_0$(Au) (GPa) & $E_\mathrm{coh}$(Hg) (eV) & $V$($\alpha$-Hg) (\AA$^3$/atom)\\",
         r"\midrule"]
for m in ["CHGNet", "MACE"]:
    r = mb.loc[m]
    lines.append(f"{NAME[m]} & {r.a0_A:.3f} & {r.B0_GPa:.0f} & {r.Ecoh_Hg_model:.3f} & {r.Hg_bulk_V_per_atom:.1f}\\\\")
for m in ["PBE", "PBE-D3"]:
    if m in db.index:
        r = db.loc[m]
        v = np.nan
        hb = os.path.join(HERE, "dft", "runs", "prod", m, "Hg_bulk", "pw.out")
        if os.path.exists(hb):   # final volume of the vc-relax (1 atom per cell)
            vl = [l for l in open(hb, errors="ignore") if "new unit-cell volume" in l]
            v = float(vl[-1].split("(")[1].split()[0]) if vl else np.nan
        lines.append(f"{NAME[m]} & {r.a0_A:.3f} & {r.B0_GPa:.0f} & {r.Ecoh_Hg:.3f} & {f(v, 1)}\\\\")
lines += [r"\midrule", r"Experiment & 4.078 & -- & 0.67 & --\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_bulk.tex"), "w").write("\n".join(lines) + "\n")

# ---- convergence table
cp = os.path.join(RES, "dft_convergence.csv")
if os.path.exists(cp):
    c = pd.read_csv(cp)
    c = c[c.test != "bulk_Au"]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Test & Setting & $E_\mathrm{ads}$(fcc) & $E_\mathrm{ads}$(ontop) & $\Delta E_\mathrm{fcc-ontop}$\\",
             r"\midrule"]
    for (t, fn, ew, er, k, v), g in c.groupby(["test", "functional", "ecutwfc", "ecutrho", "kpts", "vacuum"],
                                               sort=False):
        q = g.set_index("quantity").value
        lab = {"cutoff": f"{ew:.0f}/{er:.0f} Ry", "ecutrho": f"{ew:.0f}/{er:.0f} Ry",
               "kmesh": f"$k$ {k}", "vacuum": f"vacuum {v:.0f} \\AA", "dipole": "dipole corr.",
               "D3": "PBE-D3(BJ)"}[t]
        lines.append(f"{t} & {lab} & {q['E_ads_fcc']:.4f} & {q['E_ads_ontop']:.4f} & "
                     f"{q['dE_fcc_minus_ontop']:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_conv.tex"), "w").write("\n".join(lines) + "\n")

print(allr[allr.scheme == "A_isolated_atom"].pivot_table(index="method", columns="site",
                                                         values="E_ads_eV")[SITES].round(3))
print("tables ->", TAB)
