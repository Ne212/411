#!/usr/bin/env python
"""
Figures for the report (written to ../report/figures and ../results/figures).
Uses mlip_*.csv and, if present, dft_results.csv. LJ (ASE defaults) is a numerical
control only and is excluded from the figures.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUTS = [os.path.join(HERE, "..", "report", "figures"), os.path.join(RES, "figures")]
for o in OUTS:
    os.makedirs(o, exist_ok=True)

# fixed categorical order (validated reference palette, slots 1-4)
COLOR = {"CHGNet": "#2a78d6", "MACE": "#eb6834", "PBE": "#1baf7a", "PBE-D3": "#eda100"}
LABEL = {"CHGNet": "CHGNet 0.3.0", "MACE": "MACE-MP-0 medium", "PBE": "DFT PBE",
         "PBE-D3": "DFT PBE-D3"}
SITES = ["ontop", "bridge", "fcc", "hcp"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name):
    for o in OUTS:
        fig.savefig(os.path.join(o, name + ".pdf"))
        fig.savefig(os.path.join(o, name + ".png"))
    plt.close(fig)


mlip = pd.read_csv(os.path.join(RES, "mlip_results.csv"))
mlip = mlip[mlip.method != "LJ"]
_cmp = os.path.join(RES, "comparison_table.csv")      # written by make_tables.py (adds scheme D)
mlip_d = pd.read_csv(_cmp) if os.path.exists(_cmp) else mlip
mlip_d = mlip_d[mlip_d.method.isin(["CHGNet", "MACE"])]
frames = [mlip[mlip.scheme == "A_isolated_atom"][["method", "site", "E_ads_eV"]]]
dft_path = os.path.join(RES, "dft_results.csv")
if os.path.exists(dft_path):
    dft = pd.read_csv(dft_path)
    frames.append(dft[dft.scheme == "A_isolated_atom"][["method", "site", "E_ads_eV"]])
data = pd.concat(frames)
methods = [m for m in COLOR if m in set(data.method)]

# ---- Fig 1: E_ads per site, scheme A, all methods
fig, ax = plt.subplots(figsize=(6.2, 3.0))
w = 0.8 / len(methods)
x = np.arange(len(SITES))
for k, m in enumerate(methods):
    v = [data[(data.method == m) & (data.site == s)].E_ads_eV.mean() for s in SITES]
    ax.bar(x + (k - (len(methods) - 1) / 2) * w, v, w - 0.03, color=COLOR[m], label=LABEL[m])
ax.set_xticks(x, SITES)
ax.set_ylabel(r"$E_\mathrm{ads}$ (eV), scheme A")
ax.axhline(0, color=INK2, lw=0.8)
# experiment: -98 +- 3 kJ/mol (Soverna 2005) and the lower limit 67 kJ/mol (Lens 2018)
EXP, EXP_ERR, LIM = -98 / 96.485, 3 / 96.485, -67 / 96.485
ax.axhspan(EXP - EXP_ERR, EXP + EXP_ERR, color=INK2, alpha=0.18, lw=0)
ax.axhline(EXP, color=INK, lw=1.0, label=r"exp. $-\Delta H_\mathrm{ads}$ = 98$\pm$3 kJ/mol")
ax.axhline(LIM, color=INK2, lw=0.9, ls=":", label="exp. lower limit, 67 kJ/mol")
ax.set_ylim(min(EXP - 0.08, ax.get_ylim()[0]), 0.02)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
save(fig, "fig_eads_sites")

# ---- Fig 2: scheme differences per MLIP (B-A and C-A), meV
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=False)
for ax, scheme, title in [(axs[0], "B_far_above_slab", "B $-$ A (far-away $-$ isolated atom)"),
                          (axs[1], "D_bulkHg_plus_Ecoh_PBE", r"D $-$ A (bulk Hg + $E_\mathrm{coh}^\mathrm{PBE}$ $-$ isolated atom)")]:
    mm = ["CHGNet", "MACE"]
    w = 0.38
    for k, m in enumerate(mm):
        d = mlip_d[mlip_d.method == m].pivot_table(index="site", columns="scheme", values="E_ads_eV")
        diff = [(d.loc[s, scheme] - d.loc[s, "A_isolated_atom"]) * 1000 for s in SITES]
        ax.bar(x + (k - 0.5) * w, diff, w - 0.03, color=COLOR[m], label=LABEL[m])
        if scheme.startswith("D"):      # constant per model by construction: label once
            ax.text(x[0] + (k - 0.5) * w, diff[0] + 8, f"{diff[0]:.0f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
        else:
            maxabs = max(maxabs, max(abs(v) for v in diff)) if "maxabs" in dir() else max(abs(v) for v in diff)
    ax.set_xticks(x, SITES)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_ylabel(r"$\Delta E_\mathrm{ads}$ (meV)")
axs[0].set_ylim(-1, 1)
axs[0].text(1.5, -0.55, (f"all |B$-$A| < 0.1 meV" if maxabs < 0.1 else f"all |B$-$A| $\\leq$ {maxabs:.1f} meV") + "\n(no neighbour within the 6 Å cutoff)",
            ha="center", fontsize=8, color=INK2)
axs[0].legend(loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "fig_scheme_difference")

# ---- Fig 3: rigid dissociation scan above fcc site
scan = pd.read_csv(os.path.join(RES, "mlip_scan.csv"))
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for m in ["CHGNet", "MACE"]:
    s = scan[scan.method == m]
    ax.plot(s.height_A, s.E_minus_slab_minus_atom, color=COLOR[m], lw=2, label=LABEL[m])
dscan = os.path.join(RES, "dft_scan.csv")
if os.path.exists(dscan):
    ds = pd.read_csv(dscan)
    for m in ["PBE", "PBE-D3"]:
        s = ds[ds.method == m]
        if len(s):
            ax.plot(s.height_A, s.E_minus_slab_minus_atom, "o-", ms=4, color=COLOR[m],
                    lw=1.5, label=LABEL[m])
ax.axvline(6.0, color=INK2, lw=0.8, ls="--")
ax.text(6.08, ax.get_ylim()[0] * 0.15, "MLIP cutoff 6 Å", color=INK2, fontsize=8)
ax.axvline(10.0, color=INK2, lw=0.8, ls=":")
ax.text(10.08, ax.get_ylim()[0] * 0.15, "scheme B", color=INK2, fontsize=8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("Hg height above top Au layer (Å), fcc site, rigid slab")
ax.set_ylabel(r"$E - E_\mathrm{slab} - E_\mathrm{Hg}^\mathrm{atom}$ (eV)")
ax.legend(loc="lower right")
save(fig, "fig_dissociation_scan")
print("figures written to", OUTS)
