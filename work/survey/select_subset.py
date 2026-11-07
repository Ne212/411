"""
Talk subset chosen by fixed rules (written down before applying them; note: after Stages 2-3 were seen).
R1 context      : models used in project 411 or in Miro's example scripts: CHGNet-0.3.0, MACE-MP-0-medium,
                  ALIGNN-FF-mptrj.
R2 user decision: UMA-s-1p2p1-omat is in; TensorNet-MatPES-PBE is NOT in (shown only as an extra example in the
                  free-atom figure).
R3 training data: each group in GROUPS appears at least once.
R4 architecture : within a group, prefer a model whose architecture class is not yet in the subset;
                  ties (or no new class) -> lowest MAE (scheme D, relaxed) on the Steckel main set.
R4b OC20 group  : prefer the OC20 head of a network already in the subset (isolates the functional effect).
R5 fine-tuning  : the subset must contain each fine-tune support level in FT (checked, not used for picking).
R6 size         : at most 8.
Output: results/talk_subset.csv (model + the rule that selected it)
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "..", "results")
# group of the training data of the head used; architecture class; fine-tune support level (mlip_inventory.md)
META = {
    "CHGNet-0.3.0": ("MPtrj", "invariant GNN, 3-body", "Python API"),
    "MACE-MP-0-medium": ("MPtrj", "MACE (equivariant ACE message passing)", "documented CLI"),
    "ALIGNN-FF-mptrj": ("MPtrj", "invariant GNN, 3-body", "continue-training only"),
    "ALIGNN-FF-omat24": ("OMat24 only", "invariant GNN, 3-body", "continue-training only"),
    "ALIGNN-FF-matpes-pbe": ("MatPES-PBE", "invariant GNN, 3-body", "continue-training only"),
    "MACE-MPA-0-medium": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "MACE (equivariant ACE message passing)", "documented CLI"),
    "MACE-OMAT-0-medium": ("OMat24 only", "MACE (equivariant ACE message passing)", "documented CLI"),
    "MACE-MH-1-omat": ("OMat24 multi-domain", "MACE (equivariant ACE message passing)", "documented CLI"),
    "MACE-MH-1-oc20": ("OC20 head", "MACE (equivariant ACE message passing)", "documented CLI"),
    "TensorNet-MatPES-PBE": ("MatPES-PBE", "Cartesian-tensor equivariant", "documented CLI"),
    "M3GNet-MatPES-PBE": ("MatPES-PBE", "invariant GNN, 3-body", "documented CLI"),
    "CHGNet-MatPES-PBE": ("MatPES-PBE", "invariant GNN, 3-body", "documented CLI"),
    "SevenNet-Omni-mpa": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "NequIP-type (e3nn)", "documented CLI"),
    "SevenNet-Omni-omat24": ("OMat24 only", "NequIP-type (e3nn)", "documented CLI"),
    "SevenNet-Omni-oc20": ("OC20 head", "NequIP-type (e3nn)", "documented CLI"),
    "SevenNet-MF-ompa-mpa": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "NequIP-type (e3nn)", "documented CLI"),
    "MatterSim-5M": ("in-house (MatterSim)", "invariant GNN, 3-body", "documented CLI"),
    "ORB-v3-cons-inf-omat": ("OMat24 only", "GNS (non-equivariant)", "documented CLI"),
    "ORB-v3-cons-inf-mpa": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "GNS (non-equivariant)", "documented CLI"),
    "UMA-s-1p2p1-omat": ("OMat24 multi-domain", "eSEN + mixture of linear experts", "documented CLI"),
    "UMA-s-1p2p1-oc20": ("OC20 head", "eSEN + mixture of linear experts", "documented CLI"),
    "GRACE-2L-OAM-L": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "graph ACE", "documented CLI"),
    "GRACE-2L-SMAX-OMAT-L": ("OMat24 only", "graph ACE", "documented CLI"),
    "NequIP-OAM-L": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "NequIP-type (e3nn)", "documented CLI"),
    "Allegro-OAM-L": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "Allegro (strictly local e3nn)", "documented CLI"),
    "PET-OAM-L": ("OMat24->sAlex+MPtrj / MPtrj+sAlex", "PET (unconstrained transformer)", "documented CLI"),
    "DPA4-Plus-OMat24": ("OMat24 only", "DeePMD DPA", "documented CLI"),
    "DPA-3.1-3M-omat24": ("OMat24 only", "DeePMD DPA", "documented CLI"),
    "DPA-3.1-3M-mptrj": ("MPtrj", "DeePMD DPA", "documented CLI"),
    "DPA-3.1-3M-oc20m": ("OC20 head", "DeePMD DPA", "documented CLI"),
}
GROUPS = ["MPtrj", "OMat24->sAlex+MPtrj / MPtrj+sAlex", "OMat24 only", "OMat24 multi-domain", "MatPES-PBE", "OC20 head"]
FT = ["documented CLI", "Python API", "continue-training only"]
EXCLUDE = {"TensorNet-MatPES-PBE"}                                     # R2

M = pd.read_csv(os.path.join(RES, "published_mae.csv"))
mae = M[(M.subset == "Steckel main (30, 1 Hg/cell)") & (M["mode"] == "relaxed") & (M.scheme == "D")].set_index("model").MAE
sel = [(m, "R1 context") for m in ("CHGNet-0.3.0", "MACE-MP-0-medium", "ALIGNN-FF-mptrj")]
sel.append(("UMA-s-1p2p1-omat", "R2 user decision"))
for g in GROUPS:
    names = [m for m, _ in sel]
    if any(META[m][0] == g for m in names):
        continue
    cand = [m for m in META if META[m][0] == g and m not in EXCLUDE and m in mae.index]
    archs = {META[m][1] for m in names}
    if g == "OC20 head":
        net = lambda m: m.rsplit("-", 1)[0]          # same checkpoint, different head/task (e.g. UMA-s-1p2p1-*)
        paired = [m for m in cand if any(net(n) == net(m) for n in names)]
        pick, why = (min(paired, key=mae.get), "R3+R4b OC20 head of a network in the subset") if paired else \
                    (min(cand, key=mae.get), "R3 lowest MAE_D")
    else:
        new = [m for m in cand if META[m][1] not in archs]
        pick, why = (min(new, key=mae.get), "R3+R4 new architecture, lowest MAE_D") if new else \
                    (min(cand, key=mae.get), "R3 no new architecture available, lowest MAE_D")
    sel.append((pick, f"{why} in group '{g}'"))
assert len(sel) <= 8, "R6 violated"
names = [m for m, _ in sel]
missing_ft = [f for f in FT if not any(META[m][2] == f for m in names)]
out = pd.DataFrame([dict(model=m, rule=r, training_group=META[m][0], architecture=META[m][1], fine_tune=META[m][2],
                         MAE_D_Steckel_main=round(float(mae[m]), 3)) for m, r in sel])
out.to_csv(os.path.join(RES, "talk_subset.csv"), index=False)
print(out.to_string(index=False)); print("R5 fine-tune levels missing:", missing_ft or "none")
