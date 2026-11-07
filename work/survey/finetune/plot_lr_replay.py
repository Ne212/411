"""Talk figure for message 3: MACE-MP-0 foundation vs (a) lr 1e-3/200 ep, (b) lr 1e-4/15 ep, (c) replay lr 1e-4/15 ep.
Single seed, single split; data: results/finetune/ft_lr_replay_comparison.csv"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results")
c = pd.read_csv(os.path.join(RES, "finetune", "ft_lr_replay_comparison.csv"), index_col=0)
labels = ["foundation", "(a) lr 1e-3\n200 ep", "(b) lr 1e-4\n15 ep", "(c) replay\nlr 1e-4, 15 ep"]
cols = ["#8a8984", "#eb6834", "#eda100", "#2a78d6"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2})
panels = [("E_ads ontop (held out) − QE (eV)", "E_ads ontop, held out\nmodel − QE (eV)"),
          ("free-atom error E_coh(model) − E_coh(QE) (eV)", "free-atom error (eV)"),
          ("Steckel, Ag/Cu/Ni/Pd/Pt (25): MAE A (eV)", "Steckel Ag/Cu/Ni/Pd/Pt\nMAE (eV), never trained"),
          ("max |Δa0| Ag/Cu/Ni/Pd/Pt vs foundation (Å)", "drift of other metals\nmax |Δa0| (Å)")]
fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
for ax, (col, title) in zip(axes, panels):
    v = c[col].astype(float).values
    for i, x in enumerate(v):
        if np.isnan(x):
            ax.text(i, 0, "lost*", rotation=90, ha="center", va="bottom", fontsize=7.5, color=cols[i])
        else:
            ax.bar(i, x, width=0.8, color=cols[i], edgecolor=SURF)
            ax.text(i, x, f"{x:+.2f}" if "error" in col or "−" in col else f"{x:.2f}",
                    ha="center", va="bottom" if x >= 0 else "top", fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK, lw=0.7); ax.set_xticks(range(4), labels, fontsize=6.5)
    ax.set_title(title, fontsize=8, color=INK); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.suptitle("MACE-MP-0 fine-tuned on 84 QE frames + free Hg atom: learning rate vs replay (one seed, one split)",
             fontsize=9, x=0.01, ha="left", color=INK)
fig.text(0.01, -0.04, "* lost: flat E(a) for Ag, Cu, Ni, Pd, Pt after (a). (c) replays 2000 random MPtrj structures; "
         "mace-torch also triplicated our data in (c) (real/replay ratio < 0.1), so (c) made more updates on it than (b).",
         fontsize=6.8, color=INK2, ha="left", va="top")
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, "figures", f"finetune_lr_replay.{ext}"), dpi=200, bbox_inches="tight")
