"""
Stage 2: collect results/hg_au/<model>/ into one table, add the 411 DFT references, make figures.
Outputs
  results/hg_au_all_models.csv     one row per model x site x scheme (+ PBE, PBE-D3 rows)
  results/hg_au_model_summary.csv  one row per model: a0, atom error, MAEs vs PBE / PBE-D3, site differences
  results/figures/survey_*.png|pdf
Runs with any venv that has pandas + matplotlib (e.g. venvs/mace).
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
SITES = ["ontop", "bridge", "fcc", "hcp"]

# training data of the checkpoint/head actually used, and the DFT level of that data (see mlip_inventory.md)
META = {
    "MACE-MP-0-medium": ("MACE", "MPtrj", "PBE(+U)"), "MACE-MPA-0-medium": ("MACE", "MPtrj+sAlex", "PBE(+U)"),
    "MACE-OMAT-0-medium": ("MACE", "OMat24", "PBE(+U)"), "MACE-MH-1-omat": ("MACE", "OMat24 (MH-1 head)", "PBE(+U)"),
    "MACE-MH-1-oc20": ("MACE", "OC20 (MH-1 head)", "OC20 head: RPBE? (not verified)"),
    "CHGNet-0.3.0": ("CHGNet", "MPtrj (MP2020-corrected)", "PBE(+U)"),
    "TensorNet-MatPES-PBE": ("MatGL", "MatPES-PBE", "PBE"), "M3GNet-MatPES-PBE": ("MatGL", "MatPES-PBE", "PBE"),
    "CHGNet-MatPES-PBE": ("MatGL", "MatPES-PBE", "PBE"),
    "ALIGNN-FF-mptrj": ("ALIGNN", "MPtrj", "PBE(+U)"), "ALIGNN-FF-omat24": ("ALIGNN", "OMat24 1M", "PBE(+U)"),
    "ALIGNN-FF-matpes-pbe": ("ALIGNN", "MatPES-PBE", "PBE"),
    "SevenNet-Omni-mpa": ("SevenNet", "MPtrj+sAlex (Omni task)", "PBE(+U)"),
    "SevenNet-Omni-omat24": ("SevenNet", "OMat24 (Omni task)", "PBE(+U)"),
    "SevenNet-Omni-oc20": ("SevenNet", "OC20 (Omni task)", "RPBE"),
    "SevenNet-MF-ompa-mpa": ("SevenNet", "MPtrj+sAlex (MF task)", "PBE(+U)"),
    "MatterSim-5M": ("MatterSim", "MatterSim in-house", "not verified"),
    "ORB-v3-cons-inf-omat": ("ORB", "OMat24", "PBE(+U)"), "ORB-v3-cons-inf-mpa": ("ORB", "MPtrj+Alexandria", "PBE(+U)"),
    "UMA-s-1p2p1-omat": ("UMA", "OMat24 (task)", "PBE(+U)"), "UMA-s-1p2p1-oc20": ("UMA", "OC20 (task)", "RPBE"),
    "GRACE-2L-OAM-L": ("GRACE", "OMat24->sAlex+MPtrj", "PBE(+U)"),
    "GRACE-2L-SMAX-OMAT-L": ("GRACE", "SMAX+OMat24", "PBE(+U)"),
    "NequIP-OAM-L": ("NequIP", "OMat24->sAlex+MPtrj", "PBE(+U)"),
    "Allegro-OAM-L": ("Allegro", "OMat24->sAlex+MPtrj", "PBE(+U)"),
    "PET-OAM-L": ("PET", "OMat24->sAlex+MPtrj", "PBE(+U)"),
    "DPA4-Plus-OMat24": ("DPA", "OMat24", "PBE(+U)"),
    "DPA-3.1-3M-omat24": ("DPA", "OMat24 (head)", "PBE(+U)"), "DPA-3.1-3M-mptrj": ("DPA", "MPtrj (head)", "PBE(+U)"),
    "DPA-3.1-3M-oc20m": ("DPA", "OC20M (head)", "RPBE"),
    "DPA-3.1-3M-AgAu-PBED3": ("DPA", "Ag/Au only (head, no Hg)", "PBE-D3"),
}
FLAGGED = {"DPA-3.1-3M-AgAu-PBED3": "head trained on Ag/Au only: no Hg in its data (diagnostic only)"}


def group(key):
    tr = META[key][1]
    if "OC20" in tr: return "OC20 head/task (surfaces)"
    if "MatPES" in tr: return "MatPES-PBE"
    if "OMat24" in tr or "SMAX" in tr: return "OMat24-based"
    if key in FLAGGED: return "flagged"
    return "MPtrj / MP-like"


# ------------------------------------------------------------------ collect
rows, summ = [], []
for d in sorted(glob.glob(os.path.join(RES, "hg_au", "*", "results.csv"))):
    key = os.path.basename(os.path.dirname(d))
    if "-ft-" in key:          # Stage 4 fine-tuned models are reported separately (finetune/ft_analysis.py)
        continue
    r = pd.read_csv(d); b = pd.read_csv(os.path.join(os.path.dirname(d), "bulk.csv")).iloc[0]
    fam, tr, fn = META[key]
    r.insert(1, "family", fam); r.insert(2, "training_data", tr); r.insert(3, "training_functional", fn)
    r["Ecoh_Hg_model"] = b["Ecoh_Hg_model"]; r["atom_error_vs_PBE"] = b["atom_error_vs_PBE"]
    r["flag"] = FLAGGED.get(key, "")
    # largest Au displacement w.r.t. the relaxed clean slab (detects surface reconstruction)
    from ase.io import read
    sd = os.path.join(os.path.dirname(d), "structures")
    slab = read(os.path.join(sd, f"{key}_slab.xyz"))
    disp = {st: float(np.linalg.norm(read(os.path.join(sd, f"{key}_{st}.xyz")).positions[:len(slab)]
                                     - slab.positions, axis=1).max()) for st in SITES}
    r["max_Au_disp_A"] = r.site.map(disp)
    rows.append(r)

dft = pd.read_csv(os.path.join(RES, "dft_results.csv"))
dref = pd.read_csv(os.path.join(RES, "dft_bulk_refs.csv")).set_index("functional")
dft_rows = []
for fn, g in dft.groupby("method"):
    for _, x in g.iterrows():
        dft_rows.append(dict(method=fn, family="DFT (QE, project 411)", training_data="-", training_functional=fn,
                             a0_A=x.a0_A, site=x.site, scheme=x.scheme, E_ads_eV=x.E_ads_eV, E_total_eV=x.E_total_eV,
                             E_ref_eV=x.E_ref_eV, height_A=x.height_A, d_AuHg_min_A=x.d_AuHg_min_A,
                             converged=x.status == "OK", n_steps=x.n_ionic, Ecoh_Hg_model=dref.loc[fn, "Ecoh_Hg"],
                             atom_error_vs_PBE=dref.loc[fn, "Ecoh_Hg"] - dref.loc["PBE", "Ecoh_Hg"], flag=""))
        if x.scheme == "A_isolated_atom":   # for DFT, scheme D (own bulk Hg + PBE Ecoh) == A only for PBE
            e_d = x.E_total_eV - (dref.loc[fn, "E_slab"] + dref.loc[fn, "E_Hg_bulk_per_atom"] + dref.loc["PBE", "Ecoh_Hg"])
            dft_rows.append(dict(dft_rows[-1], scheme="D_bulkHg_plus_Ecoh_PBE", E_ads_eV=e_d, E_ref_eV=np.nan))
all_df = pd.concat(rows + [pd.DataFrame(dft_rows)], ignore_index=True)
all_df.to_csv(os.path.join(RES, "hg_au_all_models.csv"), index=False, float_format="%.6f")

A = all_df[all_df.scheme == "A_isolated_atom"].pivot(index="method", columns="site", values="E_ads_eV")[SITES]
D = all_df[all_df.scheme == "D_bulkHg_plus_Ecoh_PBE"].pivot(index="method", columns="site", values="E_ads_eV")[SITES]
for key in A.index:
    if key in ("PBE", "PBE-D3"): continue
    b = pd.read_csv(os.path.join(RES, "hg_au", key, "bulk.csv")).iloc[0]
    sub = all_df[(all_df.method == key)]
    s = dict(model=key, family=META[key][0], training_data=META[key][1], training_functional=META[key][2],
             group=group(key), flag=FLAGGED.get(key, ""), a0_A=b.a0_A, a0_err_vs_PBE_A=b.a0_A - 4.150305,
             B0_GPa=b.B0_GPa, E_Hg_atom=b.E_Hg_atom, Ecoh_Hg_model=b.Ecoh_Hg_model,
             atom_error_vs_PBE=b.atom_error_vs_PBE, Hg_bulk_converged=b.Hg_bulk_converged,
             # locality test: Hg 10 A above the slab is outside every cutoff, so for a strictly local
             # model E(far) = E(slab) + E(atom) (QE-PBE: -0.0002 eV)
             nonlocality_far_minus_slab_minus_atom=b.E_far_minus_slab - b.E_Hg_atom,
             all_relax_converged=bool(sub.converged.all()), wall_time_s=b.get("wall_time_s", np.nan),
             most_stable_site=A.loc[key].idxmin(), max_Au_disp_A=float(sub.max_Au_disp_A.max()),
             n_unconverged=int((~sub[sub.scheme == "A_isolated_atom"].converged.astype(bool)).sum()))
    for ref in ("PBE", "PBE-D3"):
        s[f"MAE_A_vs_{ref}"] = float((A.loc[key] - A.loc[ref]).abs().mean())
        s[f"MAE_D_vs_{ref}"] = float((D.loc[key] - D.loc[ref]).abs().mean())
    # reference-free: site energies relative to fcc
    rel = A.loc[key] - A.loc[key, "fcc"]; relp = A.loc["PBE"] - A.loc["PBE", "fcc"]
    for st in ("ontop", "bridge", "hcp"):
        s[f"dE_{st}_minus_fcc"] = float(rel[st])
    s["MAE_sitediff_vs_PBE"] = float((rel - relp)[["ontop", "bridge", "hcp"]].abs().mean())
    summ.append(s)
S = pd.DataFrame(summ).sort_values("atom_error_vs_PBE")
S.to_csv(os.path.join(RES, "hg_au_model_summary.csv"), index=False, float_format="%.4f")

# ------------------------------------------------------------------ figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # reference palette, fixed order
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
GROUPS = ["MPtrj / MP-like", "OMat24-based", "MatPES-PBE", "OC20 head/task (surfaces)", "flagged"]
GCOL = dict(zip(GROUPS, CAT))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


main = S[S.flag == ""]
order = list(main.sort_values("atom_error_vs_PBE").model)

# Fig 1: isolated-atom error per model
fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(order) + 1.2))
y = np.arange(len(order))
for i, m in enumerate(order):
    row = main.set_index("model").loc[m]
    ax.barh(i, row.atom_error_vs_PBE, height=0.62, color=GCOL[row.group], edgecolor=SURF, linewidth=1)
    ax.text(row.atom_error_vs_PBE + (0.01 if row.atom_error_vs_PBE >= 0 else -0.01), i,
            f"{row.atom_error_vs_PBE:+.2f}", va="center", ha="left" if row.atom_error_vs_PBE >= 0 else "right",
            fontsize=7, color=INK2)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(y, order, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("Free Hg atom error  E$_{coh}$(model) − E$_{coh}$(PBE)  [eV]")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in GROUPS[:4]]
ax.legend(handles, GROUPS[:4], fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(main.atom_error_vs_PBE.min() - 0.12, main.atom_error_vs_PBE.max() + 0.12)
ax.set_title("Hg cohesive-energy error vs QE-PBE (E$_{coh}$,PBE = %.3f eV)" % dref.loc["PBE", "Ecoh_Hg"],
             fontsize=9, color=INK, loc="left")
save(fig, "survey_atom_error")

# Fig 1b: talk subset (results/talk_subset.csv) + TensorNet as an explicitly marked extra example
if os.path.exists(os.path.join(RES, "talk_subset.csv")):
    talk = list(pd.read_csv(os.path.join(RES, "talk_subset.csv")).model)
    extra = "TensorNet-MatPES-PBE"
    sub = main.set_index("model").loc[talk + [extra]].sort_values("atom_error_vs_PBE")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    yy = np.arange(len(sub))
    for i, (m, row) in enumerate(sub.iterrows()):
        ex = m == extra
        ax.barh(i, row.atom_error_vs_PBE, height=0.6, color=SURF if ex else GCOL[row.group],
                edgecolor=GCOL[row.group], linewidth=1.2, hatch="////" if ex else None)
        v = row.atom_error_vs_PBE
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yy, [m + ("  (extra example)" if m == extra else "") for m in sub.index], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(-0.75, 0.6)
    ax.set_xlabel("Free Hg atom error  E$_{coh}$(model) − E$_{coh}$(PBE)  [eV]")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    gs = [g for g in GROUPS[:4] if (sub.group == g).any()]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in gs], gs, fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Free-atom error, talk subset", fontsize=9, color=INK, loc="left")
    fig.text(0.01, -0.02, "Hatched: TensorNet-MatPES-PBE is not in the talk subset; it is added only because it has the largest "
             "free-atom error of all 31 models (−0.61 eV)\nwhile its error after removing the free atom (scheme D) is small "
             "(0.054 eV on Steckel 2008), the clearest illustration of the free-atom effect.",
             fontsize=6.8, color=INK2, ha="left", va="top")
    save(fig, "survey_atom_error_talk")

# Fig 2: E_ads at four sites, scheme A and scheme D, models in the same order, PBE / PBE-D3 bands
MK = {"ontop": ("o", CAT[0]), "bridge": ("s", CAT[1]), "fcc": ("^", CAT[2]), "hcp": ("D", CAT[3])}
DY = {"ontop": -0.27, "bridge": -0.09, "fcc": 0.09, "hcp": 0.27}       # vertical dodge: fcc ~ hcp would hide
labels = [m + (" *" if m.startswith("UMA") else "") for m in order]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.26 * len(order) + 1.8), sharey=True, sharex=True)
for ax, (tab, title) in zip(axes, [(A, "A: E(slab+Hg) − E(slab) − E(Hg atom, model)"),
                                   (D, "D: free atom → bulk α-Hg(model) + E$_{coh}$(PBE)")]):
    for ref, ls in (("PBE", "-"), ("PBE-D3", "--")):
        lo, hi = tab.loc[ref].min(), tab.loc[ref].max()
        ax.axvspan(lo, hi, color=INK2, alpha=0.10 if ref == "PBE" else 0.05, lw=0)
        ax.axvline(tab.loc[ref, "fcc"], color=INK2, lw=1, ls=ls, label=f"QE {ref} (fcc)")
    for st in SITES:
        mk, c = MK[st]
        ax.scatter(tab.loc[order, st], y + DY[st], marker=mk, s=20, color=c, edgecolor=SURF, linewidth=0.5,
                   label=st, zorder=3)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("E$_{ads}$ [eV]")
axes[0].set_yticks(y, labels, fontsize=7.5); axes[0].invert_yaxis()
axes[1].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Hg/Au(111), 411 protocol, each model at its own a$_0$ (band: QE range over the four sites)",
             fontsize=9, color=INK, x=0.01, y=0.995, ha="left")
fig.text(0.01, -0.005, "* UMA is not local: E(Hg 10 Å above slab) − E(slab) − E(Hg atom) = −0.13 eV, so scheme A "
         "differs from scheme B by 0.13 eV; all other models: |Δ| < 0.01 eV. Models sorted by free-atom error.",
         fontsize=7, color=INK2, ha="left", va="top")
fig.tight_layout()
save(fig, "survey_sites")

# Fig 3: reference-free site differences (relative to fcc) vs PBE
fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(order) + 1.2))
for st, (mk, c) in [("ontop", MK["ontop"]), ("bridge", MK["bridge"]), ("hcp", MK["hcp"])]:
    ax.scatter(main.set_index("model").loc[order, f"dE_{st}_minus_fcc"], y, marker=mk, s=22, color=c,
               edgecolor=SURF, linewidth=0.6, label=f"{st} − fcc", zorder=3)
    p = A.loc["PBE", st] - A.loc["PBE", "fcc"]
    ax.axvline(p, color=c, lw=1, ls=":")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(y, order, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("E$_{ads}$(site) − E$_{ads}$(fcc)  [eV]   (dotted: QE-PBE)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Site differences: independent of the gas-phase reference", fontsize=9, color=INK, loc="left")
save(fig, "survey_site_differences")

print(S[["model", "nonlocality_far_minus_slab_minus_atom", "max_Au_disp_A", "n_unconverged"]].round(4).to_string(index=False))
print(S[["model", "a0_A", "atom_error_vs_PBE", "MAE_A_vs_PBE", "MAE_D_vs_PBE", "MAE_A_vs_PBE-D3",
         "MAE_sitediff_vs_PBE", "most_stable_site", "all_relax_converged"]].round(3).to_string(index=False))
print("missing:", sorted(set(META) - set(S.model)))
