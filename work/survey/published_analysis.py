"""
Stage 3 analysis. Errors are reported PER SOURCE, never pooled across sources/functionals:
  * Steckel 2008 (VASP-PBE): main set = 30 systems with 1 Hg per cell (per-Hg vs per-cell irrelevant),
    reported with and without Ni; the 14 systems with >1 Hg per supercell (normalisation not stated in the paper)
    are reported separately.
  * QE-PBE, project 411 (Hg/Au(111) fcc, 0.111 ML): separate single-system error, from Stage 2.
Metrics per model, mode (source geometry / relaxed) and scheme (A: isolated Hg atom; D: bulk Hg + E_coh,PBE):
  MAE, mean signed error (ME), MAE after removing the model's own mean offset ("centred MAE", i.e. errors in
  trends only), Spearman rank correlation.
Outputs: results/published_benchmark.csv (wide), results/published_benchmark_long.csv, results/published_mae.csv,
         results/published_mae.md, results/figures/published_*.png|pdf
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results"); FIG = os.path.join(RES, "figures")
SRC = pd.read_csv(os.path.join(RES, "published", "steckel2008_systems.csv"))
L = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "published", "models", "*.csv")))
               if "-ft-" not in os.path.basename(f)],          # Stage 4 fine-tuned models: reported separately
              ignore_index=True)
summ2 = pd.read_csv(os.path.join(RES, "hg_au_model_summary.csv")).set_index("model")
L["err_A"] = L.E_ads_A_eV - L.E_ads_ref_eV
L["err_D"] = L.E_ads_D_eV - L.E_ads_ref_eV
L.to_csv(os.path.join(RES, "published_benchmark_long.csv"), index=False, float_format="%.5f")

# wide table as specified: system, source, DOI, functional, DFT value, one column per model (relaxed, scheme A)
W = SRC[["id", "source", "doi", "functional", "E_ads_definition", "metal", "facet", "overlayer", "coverage_ML",
         "E_ads_eV", "in_main_MAE", "normalisation", "Ni_flag"]].rename(columns={"id": "system", "E_ads_eV": "E_ads_DFT_eV"})
W["functional"] = "PBE"
piv = L[L["mode"] == "relaxed"].pivot(index="system", columns="model", values="E_ads_A_eV")
W = W.merge(piv, left_on="system", right_index=True, how="left")
W.to_csv(os.path.join(RES, "published_benchmark.csv"), index=False, float_format="%.4f")

SUBSETS = {"Steckel main (30, 1 Hg/cell)": lambda d: d.in_main_MAE,
           "Steckel main without Ni (25)": lambda d: d.in_main_MAE & ~d.Ni_flag,
           "Steckel >1 Hg/cell, normalisation unverified (14)": lambda d: ~d.in_main_MAE}
rows = []
for (m, mode), g in L.groupby(["model", "mode"]):
    for sname, sel in SUBSETS.items():
        h = g[sel(g)]
        if len(h) == 0:
            continue
        for sch in ("A", "D"):
            e = h[f"err_{sch}"]; v = h[f"E_ads_{sch}_eV"]
            rows.append(dict(model=m, mode=mode, subset=sname, scheme=sch, n=len(h), MAE=e.abs().mean(), ME=e.mean(),
                             MAE_centred=(e - e.mean()).abs().mean(),
                             spearman=spearmanr(v, h.E_ads_ref_eV).statistic if len(h) > 2 else np.nan))
M = pd.DataFrame(rows)
# separate source: QE-PBE 411 (single system Hg/Au(111) fcc 0.111 ML) from Stage 2
qe = []
for m in M.model.unique():
    if m in summ2.index:
        a = pd.read_csv(os.path.join(RES, "hg_au", m, "results.csv")).set_index(["site", "scheme"])
        ref = pd.read_csv(os.path.join(RES, "dft_results.csv")).query("method=='PBE' and site=='fcc'").set_index("scheme")
        for sch, key in (("A", "A_isolated_atom"), ("D", "D_bulkHg_plus_Ecoh_PBE")):
            refv = ref.loc["A_isolated_atom", "E_ads_eV"]   # PBE: D == A by construction
            e = a.loc[("fcc", key), "E_ads_eV"] - refv
            qe.append(dict(model=m, mode="relaxed", subset="QE-PBE 411, Hg/Au(111) fcc 0.111 ML (1)", scheme=sch, n=1,
                           MAE=abs(e), ME=e, MAE_centred=np.nan, spearman=np.nan))
M = pd.concat([M, pd.DataFrame(qe)], ignore_index=True)
FLAG = {"DPA-3.1-3M-AgAu-PBED3": "flagged: head trained on Ag/Au only (no Hg, Cu, Ni, Pd, Pt)"}
M["flag"] = M.model.map(FLAG).fillna("")
unconv = L[(L["mode"] == "relaxed") & ~L.converged.astype(bool)].groupby("model").system.apply(lambda x: ", ".join(x))
M["unconverged_relaxations"] = M.model.map(unconv).fillna("")
M.to_csv(os.path.join(RES, "published_mae.csv"), index=False, float_format="%.4f")

# markdown summary: main set, relaxed, schemes A and D side by side + other subsets
def tab(subset, mode="relaxed"):
    t = M[(M.subset == subset) & (M["mode"] == mode)].pivot(index="model", columns="scheme",
                                                              values=["MAE", "ME", "MAE_centred", "spearman"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.sort_values("MAE_D")
    t.index = [i + (" (flagged)" if i in FLAG else "") for i in t.index]
    return t
with open(os.path.join(RES, "published_mae.md"), "w") as fh:
    fh.write("# Stage 3 errors vs published DFT (eV); one table per source/subset, never pooled\n\n"
             "Source: Steckel, PRB 77, 115412 (2008), doi:10.1103/PhysRevB.77.115412, VASP-PBE. "
             "A = vs isolated Hg atom (as in the paper); D = isolated atom replaced by bulk α-Hg(model) + "
             "E_coh,PBE (QE). ME = mean signed error (negative = overbinding). MAE_centred = MAE after removing the "
             "model's mean offset. Mode 'relaxed' unless stated. (flagged) = DPA-3.1 AgAu-PBED3 head, trained on "
             "Ag/Au only. Unconverged relaxations (included, marked in published_mae.csv): "
             + "; ".join(f"{k}: {v}" for k, v in unconv.items()) + ".\n\n")
    for sname in list(SUBSETS) + ["QE-PBE 411, Hg/Au(111) fcc 0.111 ML (1)"]:
        fh.write(f"## {sname}\n\n" + tab(sname).round(3).to_markdown() + "\n\n")
    fh.write("## Steckel main (30), source geometry (single points at the paper's M–Hg distance)\n\n"
             + tab("Steckel main (30, 1 Hg/cell)", "source").round(3).to_markdown() + "\n")

# ------------------------------------------------------------------ figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Fig 1: MAE per model, main set, relaxed, A vs D (two marks on one axis, same unit)
t = tab("Steckel main (30, 1 Hg/cell)")
t.index = [i.replace(" (flagged)", "") for i in t.index]
t = t.drop(index=list(FLAG), errors="ignore")             # flagged head not plotted
order = list(t.sort_values("MAE_D").index)
fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(order) + 1.3))
y = np.arange(len(order))
for x0, x1, yy in zip(t.loc[order, "MAE_A"], t.loc[order, "MAE_D"], y):
    ax.plot([x0, x1], [yy, yy], color=GRID, lw=1.5, zorder=1)
ax.scatter(t.loc[order, "MAE_A"], y, s=26, color=CAT[1], label="A: vs isolated Hg atom (model)", zorder=3)
ax.scatter(t.loc[order, "MAE_D"], y, s=26, color=CAT[0], marker="D", label="D: free-atom error removed", zorder=3)
ax.set_yticks(y, order, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("MAE vs Steckel 2008 VASP-PBE, 30 systems (Hg on 6 metals, (111)+(001)) [eV]")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Published PBE benchmark: error per model (relaxed geometry)", fontsize=9, loc="left", color=INK)
save(fig, "published_mae")


def parity(models, name, title):
    n = len(models); nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.5 * nc, 2.5 * nr + 0.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    lim = (-2.0, 0.7)
    for ax, m in zip(axes, models):
        g = L[(L.model == m) & (L["mode"] == "relaxed")]
        ax.plot(lim, lim, color=INK2, lw=0.8)
        for sch, c, mk in (("A", CAT[1], "o"), ("D", CAT[0], "D")):
            for main, alpha in ((True, 1.0), (False, 0.35)):
                h = g[g.in_main_MAE == main]
                ax.scatter(h.E_ads_ref_eV, h[f"E_ads_{sch}_eV"], s=12 if main else 9, color=c, marker=mk,
                           alpha=alpha, edgecolor=SURF, linewidth=0.3,
                           label=(f"scheme {sch}" + ("" if main else ", >1 Hg/cell")) if m == models[0] else None)
        r = M[(M.model == m) & (M["mode"] == "relaxed") & (M.subset == "Steckel main (30, 1 Hg/cell)")].set_index("scheme")
        ax.text(0.04, 0.96, f"{m}\nMAE A {r.loc['A','MAE']:.2f} / D {r.loc['D','MAE']:.2f}", transform=ax.transAxes,
                fontsize=6.5, va="top", color=INK)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.supxlabel("E$_{ads}$ Steckel 2008, VASP-PBE [eV]", fontsize=8); fig.supylabel("E$_{ads}$ model [eV]", fontsize=8)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower right", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


# talk subset (reasons in results/published_notes.md)
TALK = list(pd.read_csv(os.path.join(RES, "talk_subset.csv")).model)          # rules: select_subset.py
parity(TALK, "published_parity_talk", "Hg on Ag, Au, Cu, Ni, Pd, Pt vs Steckel 2008 (VASP-PBE): talk subset")
parity(order, "published_parity_all", "Hg on Ag, Au, Cu, Ni, Pd, Pt (111)/(001): all models (faded: >1 Hg per cell, not in MAE)")
print(tab("Steckel main (30, 1 Hg/cell)").round(3).to_string())
print(tab("Steckel main without Ni (25)")[["MAE_A", "MAE_D"]].round(3).to_string())
