"""Accept/reject grid of heavy and superheavy elements per model (results/survey_logs/element_coverage.jsonl).
No energies are shown (by design): accepted only means the code evaluates the element, not that it was trained on it."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
rows = [json.loads(l) for l in open(os.path.join(RES, "survey_logs", "element_coverage.jsonl"))]
EL = ["Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
M = np.array([[1 if r[e].startswith("accepted") else 0 for e in EL] for r in rows])
names = [r["model"] for r in rows]
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ACC, REJ = "#2a78d6", "#e4e3df"
fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(rows) + 1.6)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
for i in range(len(rows)):
    for j in range(len(EL)):
        ax.add_patch(plt.Rectangle((j, i), 0.92, 0.86, color=ACC if M[i, j] else REJ))
ax.axvline(7 - 0.04, color=INK, lw=1.2)
ax.text(3.5, -0.6, "6th-row homologues", ha="center", fontsize=8, color=INK2)
ax.text(10.5, -0.6, "superheavy (Z ≥ 112): in NO training set", ha="center", fontsize=8, color=INK)
ax.set_xlim(0, len(EL)); ax.set_ylim(len(rows), -1.1)
ax.set_xticks(np.arange(len(EL)) + 0.46, EL); ax.set_yticks(np.arange(len(rows)) + 0.43, names, fontsize=7.5)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.legend([plt.Rectangle((0, 0), 1, 1, color=ACC), plt.Rectangle((0, 0), 1, 1, color=REJ)],
          ["accepted by the code (Au–X dimer evaluates)", "rejected (error)"], loc="upper center",
          bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=7.5)
fig.text(0.01, -0.06, "Accepted ≠ trained. The element tables of the MACE models trained on MPtrj, sAlex, OMat24 and OC20 end at "
         "Z = 94 (Pu); CHGNet's embedding has 94 rows.\nNumbers that some codes return for Cn…Og come from untrained "
         "embeddings and are deliberately not shown.",
         fontsize=7, color=INK2, ha="left", va="top")
ax.set_title("Which elements does each MLIP accept?", fontsize=9, loc="left", color=INK)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, "figures", f"element_coverage.{ext}"), dpi=200, bbox_inches="tight")
