"""
Stage 4 summary: foundation vs fine-tuned (without / with the free Hg atom) for MACE-MP-0 medium and CHGNet 0.3.0.
Feasibility test on a SMALL data set (84 training frames from one QE study), not a production fine-tune.
Outputs: results/finetune/ft_summary.csv, ft_summary.md, results/figures/finetune_summary.png|pdf
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "..", "..", "results")
OUT = os.path.join(RES, "finetune"); FIG = os.path.join(RES, "figures")
MODELS = {"MACE-MP-0": ["MACE-MP-0-medium", "MACE-MP-0-ft-noatom", "MACE-MP-0-ft-atom"],
          "CHGNet": ["CHGNet-0.3.0", "CHGNet-ft-noatom", "CHGNet-ft-atom"]}
LABEL = {0: "foundation", 1: "fine-tuned, no free atom", 2: "fine-tuned, + free atom"}
TRAINED_SITES, HELD_OUT = ["bridge", "fcc", "hcp"], "ontop"
qe = pd.read_csv(os.path.join(RES, "dft_results.csv")).query("method=='PBE' and scheme=='A_isolated_atom'").set_index("site")
qeb = pd.read_csv(os.path.join(RES, "dft_bulk_refs.csv")).set_index("functional").loc["PBE"]
a0_found = {}

EXTRA = [("MACE-MP-0", "MACE-MP-0-ft-atom-lr1e-4", "(b) + free atom, no replay, lr 1e-4, 15 epochs"),
         ("MACE-MP-0", "MACE-MP-0-ft-atom-replay", "(c) + free atom, replay 2000 MPtrj, lr 1e-4, 15 epochs")]
todo = [(fam, i, k, LABEL[i]) for fam, keys in MODELS.items() for i, k in enumerate(keys)]
todo += [(fam, 9, k, lab) for fam, k, lab in EXTRA
         if os.path.exists(os.path.join(RES, "published", "models", f"{k}.csv"))]
rows = []
for fam, i, k, lab in todo:
        r = dict(family=fam, model=k, variant=lab)
        fr = pd.read_csv(os.path.join(OUT, f"frames_{k}.csv"))
        for sp in ("train", "valid", "test"):
            g = fr[fr.split == sp]
            r[f"{sp}_E_MAE_meV_atom"] = 1000 * g.dE_per_atom.abs().mean()
            r[f"{sp}_Etot_MAE_eV"] = g.dE_total.abs().mean()
            r[f"{sp}_F_MAE_meV_A"] = 1000 * g.F_MAE.mean()
        r["Hg_atom_frame_dE_eV"] = float(fr[(fr.split == "atom_group") & (fr.calc == "Hg_atom")].dE_total.iloc[0])
        h = pd.read_csv(os.path.join(RES, "hg_au", k, "results.csv"))
        h = h[h.scheme == "A_isolated_atom"].set_index("site")
        b = pd.read_csv(os.path.join(RES, "hg_au", k, "bulk.csv")).iloc[0]
        for s in ("ontop", "bridge", "fcc", "hcp"):
            r[f"Eads_{s}"] = h.loc[s, "E_ads_eV"]; r[f"err_{s}"] = h.loc[s, "E_ads_eV"] - qe.loc[s, "E_ads_eV"]
        r["err_heldout_ontop"] = r["err_ontop"]
        r["MAE_trained_sites"] = np.mean([abs(r[f"err_{s}"]) for s in TRAINED_SITES])
        r["atom_error_vs_PBE"] = b.atom_error_vs_PBE
        r["a0_Au"] = b.a0_A; r["a0_Au_err_vs_QE"] = b.a0_A - qeb.a0_A
        r["B0_Au_GPa"] = b.B0_GPa; r["B0_Au_err_vs_QE"] = b.B0_GPa - qeb.B0_GPa
        r["Ecoh_Hg_model"] = b.Ecoh_Hg_model
        # Steckel (never trained on): Au rows and the other five metals separately
        p = pd.read_csv(os.path.join(RES, "published", "models", f"{k}.csv"))
        p = p[(p["mode"] == "relaxed") & p.in_main_MAE]
        for name, sel in (("Steckel_Au", p.metal == "Au"), ("Steckel_otherMetals", p.metal != "Au"),
                          ("Steckel_otherMetals_noNi", (p.metal != "Au") & (p.metal != "Ni"))):
            q = p[sel]
            r[f"{name}_n"] = len(q)
            r[f"{name}_MAE_A"] = (q.E_ads_A_eV - q.E_ads_ref_eV).abs().mean()
            r[f"{name}_MAE_D"] = (q.E_ads_D_eV - q.E_ads_ref_eV).abs().mean()
            r[f"{name}_ME_A"] = (q.E_ads_A_eV - q.E_ads_ref_eV).mean()
        st = json.load(open(os.path.join(RES, "published", "models", k, "settings.json")))
        a0 = {m: (np.nan if v is None else v) for m, v in st["a0"].items()}
        r["metals_without_bound_lattice"] = ",".join(st.get("metals_without_bound_lattice", [])) or "none"
        if i == 0:
            a0_found[fam] = a0
        d = [abs(a0[m] - a0_found[fam][m]) for m in a0 if m != "Au" and not np.isnan(a0[m])]
        r["max_abs_change_a0_other_metals_vs_foundation_A"] = max(d) if d else np.nan
        for m in sorted(a0):
            r[f"a0_{m}"] = a0[m]
        rows.append(r)
S = pd.DataFrame(rows)
S.loc[S.model == "MACE-MP-0-ft-atom", "variant"] = "(a) + free atom, no replay, lr 1e-3, 200 epochs"
S.to_csv(os.path.join(OUT, "ft_summary.csv"), index=False, float_format="%.4f")

cols = [("test_Etot_MAE_eV", "held-out ontop frames: |ΔE| per frame (eV)"),
        ("test_F_MAE_meV_A", "held-out ontop frames: force MAE (meV/Å)"),
        ("train_F_MAE_meV_A", "training frames: force MAE (meV/Å)"),
        ("err_heldout_ontop", "E_ads ontop (held out) − QE (eV)"),
        ("MAE_trained_sites", "E_ads bridge/fcc/hcp (trained): MAE vs QE (eV)"),
        ("atom_error_vs_PBE", "free-atom error E_coh(model) − E_coh(QE) (eV)"),
        ("a0_Au_err_vs_QE", "a0(Au) − QE (Å)"), ("B0_Au_err_vs_QE", "B0(Au) − QE (GPa)"),
        ("Steckel_Au_MAE_A", "Steckel, Au rows (5): MAE A (eV)"),
        ("Steckel_otherMetals_MAE_A", "Steckel, Ag/Cu/Ni/Pd/Pt (25): MAE A (eV)"),
        ("Steckel_otherMetals_MAE_D", "Steckel, Ag/Cu/Ni/Pd/Pt (25): MAE D (eV)"),
        ("Steckel_otherMetals_noNi_MAE_A", "Steckel, Ag/Cu/Pd/Pt (20): MAE A (eV)"),
        ("max_abs_change_a0_other_metals_vs_foundation_A", "max |Δa0| Ag/Cu/Ni/Pd/Pt vs foundation (Å)"),
        ("metals_without_bound_lattice", "metals with flat / minimum-less E(a) (lost)")]
with open(os.path.join(OUT, "ft_summary.md"), "w") as fh:
    fh.write("# Stage 4: fine-tuning feasibility test (small data set: 84 training frames, one QE-PBE study)\n\n"
             "Held-out test = every frame of the ontop-site relaxation. Steckel 2008 systems were never used in "
             "training. QE references: project 411 (E_ads PBE, scheme A; a0 = %.4f Å, B0 = %.1f GPa).\n\n"
             % (qeb.a0_A, qeb.B0_GPa))
    t = S.set_index("model")[[c for c, _ in cols]].T
    t.index = [d for _, d in cols]
    fh.write(t.to_markdown(floatfmt=".4f") + "\n")
    cmp = ["MACE-MP-0-medium", "MACE-MP-0-ft-atom"] + [k for _, k, _ in EXTRA]
    cmp = [k for k in cmp if k in set(S.model)]
    fh.write("\n## MACE: learning rate vs replay (same training data incl. free Hg atom; single seed, single split)\n\n"
             "(a) is the main Stage 4 run (lr 1e-3, 200 epochs, no replay). (b) is the control: no replay, lr 1e-4, "
             "15 epochs. (c) multihead fine-tuning with replay of 2000 random MPtrj structures (mace-torch default "
             "source), lr 1e-4, 15 epochs, evaluated with the fine-tuned head 'Default'. (b) and (c) differ from (a) in lr "
             "AND number of epochs; (b) vs (c) isolates replay.\n\n")
    keep = ["err_heldout_ontop", "MAE_trained_sites", "atom_error_vs_PBE", "a0_Au_err_vs_QE", "B0_Au_err_vs_QE",
            "Steckel_Au_MAE_A", "Steckel_otherMetals_MAE_A", "Steckel_otherMetals_MAE_D", "Steckel_otherMetals_noNi_MAE_A",
            "max_abs_change_a0_other_metals_vs_foundation_A", "metals_without_bound_lattice"]
    c2 = S.set_index("model").loc[cmp, ["variant"] + keep]
    c2.columns = ["run"] + [dict(cols)[c] for c in keep]
    fh.write(c2.to_markdown(floatfmt=".3f") + "\n")
    c2.to_csv(os.path.join(OUT, "ft_lr_replay_comparison.csv"), float_format="%.4f")

# figure: small multiples, one panel per metric, 3 bars per family
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2})
panels = [("err_heldout_ontop", "E_ads ontop, held out\n(model − QE, eV)"),
          ("MAE_trained_sites", "E_ads trained sites\nMAE vs QE (eV)"),
          ("atom_error_vs_PBE", "free-atom error\n(eV)"),
          ("a0_Au_err_vs_QE", "a0(Au) − QE (Å)"),
          ("Steckel_otherMetals_MAE_A", "Steckel Ag/Cu/Ni/Pd/Pt\nMAE A (eV)"),
          ("test_F_MAE_meV_A", "held-out frames\nforce MAE (meV/Å)")]
fig, axes = plt.subplots(1, len(panels), figsize=(12, 2.8))
for ax, (c, t) in zip(axes, panels):
    for j, fam in enumerate(MODELS):
        for i, k in enumerate(MODELS[fam]):
            v = S.set_index("model").loc[k, c]
            if np.isnan(v):     # metal(s) lost: no bound fcc lattice -> no value, say so explicitly
                ax.text(j * 4 + i, 0, "lost*", rotation=90, ha="center", va="bottom", fontsize=7, color=CAT[i])
                continue
            ax.bar(j * 4 + i, v, width=0.9, color=CAT[i], edgecolor=SURF, linewidth=1,
                   label=LABEL[i] if (j == 0 and ax is axes[0]) else None)
    ax.axhline(0, color=INK, lw=0.7); ax.set_xticks([1, 5], list(MODELS)); ax.set_title(t, fontsize=7.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Fine-tuning feasibility test (84 training frames from one QE-PBE study; ontop site held out)",
             fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.text(0.01, -0.09, "* lost: after fine-tuning, MACE gives a flat E(a) for Ag, Cu, Ni, Pd and Pt (no lattice minimum), so the "
         "Steckel systems of these metals cannot be built (catastrophic forgetting).\n"
         "CHGNet '+ free atom': CHGNet's StructureData silently drops structures with an isolated atom (the free Hg atom and the "
         "'far' frame), so this run saw the same data as 'no free atom'.", fontsize=7, color=INK2, ha="left", va="top")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"finetune_summary.{ext}"), dpi=200, bbox_inches="tight")
print(open(os.path.join(OUT, "ft_summary.md")).read())
