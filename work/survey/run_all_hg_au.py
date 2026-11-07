"""
Run the 411 Hg/Au(111) protocol (hg_au_protocol.py) for every model, each in its own venv.
Resumable: a model with results/hg_au/<key>/results.csv is skipped unless --force.
Usage: python3 run_all_hg_au.py [--models K1 K2 ...] [--force]
Log: results/survey_logs/run_all_hg_au.log
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from calcs import VENV

# fast models first; DPA-3.1 (~2 s per call) last; the AgAu-PBED3 head is a flagged diagnostic
ORDER = ["ALIGNN-FF-mptrj", "ALIGNN-FF-omat24", "ALIGNN-FF-matpes-pbe",
         "CHGNet-0.3.0", "TensorNet-MatPES-PBE", "M3GNet-MatPES-PBE", "CHGNet-MatPES-PBE",
         "SevenNet-Omni-mpa", "SevenNet-Omni-omat24", "SevenNet-Omni-oc20", "SevenNet-MF-ompa-mpa",
         "MatterSim-5M", "ORB-v3-cons-inf-omat", "ORB-v3-cons-inf-mpa",
         "GRACE-2L-OAM-L", "GRACE-2L-SMAX-OMAT-L", "NequIP-OAM-L", "Allegro-OAM-L",
         "PET-OAM-L", "DPA4-Plus-OMat24",
         "MACE-MP-0-medium", "MACE-MPA-0-medium", "MACE-OMAT-0-medium", "MACE-MH-1-omat", "MACE-MH-1-oc20",
         "UMA-s-1p2p1-omat", "UMA-s-1p2p1-oc20",
         "DPA-3.1-3M-omat24", "DPA-3.1-3M-mptrj", "DPA-3.1-3M-oc20m", "DPA-3.1-3M-AgAu-PBED3"]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=ORDER)
ap.add_argument("--force", action="store_true")
ap.add_argument("--task", default="hg_au", choices=["hg_au", "published"],
                help="hg_au: hg_au_protocol.py (Stage 2); published: published_benchmark.py (Stage 3)")
args = ap.parse_args()

env = dict(os.environ, CUDA_HOME=os.path.join(ROOT, "tools", "cuda13"), PYTHONWARNINGS="ignore")
logf = open(os.path.join(ROOT, "results", "survey_logs", f"run_all_{args.task}.log"), "a")
for m in args.models:
    script, done = {"hg_au": ("hg_au_protocol.py", os.path.join(ROOT, "results", "hg_au", m, "results.csv")),
                    "published": ("published_benchmark.py",
                                  os.path.join(ROOT, "results", "published", "models", f"{m}.csv"))}[args.task]
    if os.path.exists(done) and not args.force:
        print(f"skip {m} (done)", flush=True); continue
    py = os.path.join(ROOT, "venvs", VENV[m], "bin", "python")
    t0 = time.time()
    out = os.path.join(ROOT, "results", "survey_logs", f"{args.task}_{m}.out")
    with open(out, "w") as fh:
        rc = subprocess.call([py, os.path.join(HERE, script), m], cwd=HERE, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
    msg = f"{time.strftime('%F %T')} [{args.task}] {m}: rc={rc} {time.time()-t0:.0f} s"
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
