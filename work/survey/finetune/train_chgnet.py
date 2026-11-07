"""
Fine-tune CHGNet 0.3.0 on the 411 QE data (full fine-tuning, no frozen layers, no replay).
Settings mirror the MACE run where the codes allow (train_all.sh): Adam, lr 1e-3, 200 epochs, batch 4,
force/energy loss ratio 10, seed 0. Targets: energy per atom (CHGNet convention) and forces.
Checkpoint used afterwards: bestE_* (lowest validation energy MAE). Note: Trainer.best_model is a reference to
the live model, not a copy, so the saved file is used instead.
Usage: <chgnet venv>/bin/python train_chgnet.py {noatom|atom}
"""
import glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
from chgnet.data.dataset import StructureData
from chgnet.trainer import Trainer
from torch.utils.data import DataLoader
from chgnet.data.dataset import collate_graphs

HERE = os.path.dirname(os.path.abspath(__file__)); variant = sys.argv[1]
D = os.path.join(HERE, "data", "CHGNet-0.3.0")
OUT = os.environ.get("FT_OUT", os.path.join(HERE, "runs", f"CHGNet-ft-{variant}"))   # FT_OUT/FT_EPOCHS: smoke tests only
os.makedirs(OUT, exist_ok=True)
SET = dict(lr=1e-3, epochs=int(os.environ.get("FT_EPOCHS", 200)), batch_size=4, energy_loss_ratio=1.0, force_loss_ratio=10.0, optimizer="Adam",
           scheduler="CosLR", criterion="MSE", seed=0, frozen_layers="none", replay="none")


def load(f):
    at = read(os.path.join(D, f), index=":")
    s = [AseAtomsAdaptor.get_structure(a) for a in at]
    e = [a.info["REF_energy"] / len(a) for a in at]
    fo = [a.arrays["REF_forces"].tolist() for a in at]
    return StructureData(structures=s, energies=e, forces=fo, shuffle=True)


torch.manual_seed(SET["seed"])
tr, va = load(f"train_{variant}.xyz"), load("valid.xyz")
mk = lambda ds, sh: DataLoader(ds, batch_size=SET["batch_size"], shuffle=sh, collate_fn=collate_graphs)
model = CHGNet.load()
trainer = Trainer(model=model, targets="ef", energy_loss_ratio=SET["energy_loss_ratio"],
                  force_loss_ratio=SET["force_loss_ratio"], optimizer=SET["optimizer"], scheduler=SET["scheduler"],
                  criterion=SET["criterion"], epochs=SET["epochs"], learning_rate=SET["lr"], use_device="cuda",
                  torch_seed=SET["seed"], data_seed=SET["seed"], print_freq=1000)
t0 = time.time()
trainer.train(mk(tr, True), mk(va, False), save_dir=OUT)
best = sorted(glob.glob(os.path.join(OUT, "bestE_*.pth.tar")))[-1]
json.dump(dict(settings=SET, variant=variant, n_train=len(tr), n_valid=len(va), best_checkpoint=best,
               wall_s=time.time() - t0), open(os.path.join(OUT, "train.json"), "w"), indent=1)
print("best:", best, f"{time.time()-t0:.0f} s")
