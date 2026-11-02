"""
Shared Quantum ESPRESSO settings and runner for the Hg/Au(111) DFT reference.

Every DFT calculation in this project goes through `run_pw`, so all calculations share
the same pseudopotentials, smearing, convergence thresholds and dispersion settings;
only the parameters passed explicitly (calculation type, k-points, cutoff during
convergence tests) differ, and those are written to the per-run JSON log.
"""
import json
import os
import shutil
import subprocess
import time

import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
PSEUDO_DIR = os.path.join(HERE, "pseudo")
RUN_ROOT = os.path.join(HERE, "runs")
LAUNCHERS = "/home/ne212/Çalışmalar/Çalıştır"

# SSSP 1.3.0 PBE precision (md5 checked against SSSP_1.3.0_PBE_precision.json)
PSEUDOS = {"Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",       # SG15 ONCV, NC, SR, Z=19, 50/200 Ry
           "Hg": "Hg.us.z_12.uspp.gbrv.v1.upf"}      # GBRV 1.0 USPP, SR, Z=12, 65/390 Ry

RY_BOHR_TO_EV_A = 25.71104309541616
FMAX_EV_A = 0.02                                   # same as the MLIP benchmark

BASE = {
    "control": {
        "tprnfor": True,
        "tstress": False,
        "forc_conv_thr": FMAX_EV_A / RY_BOHR_TO_EV_A,   # 7.78e-4 Ry/bohr (per component)
        "etot_conv_thr": 1.0e-5,                         # Ry
        "disk_io": "low",
        "pseudo_dir": PSEUDO_DIR,
    },
    "system": {
        "ecutwfc": 65.0,          # SSSP precision max(Au 50, Hg 65); revisited by convergence test
        "ecutrho": 390.0,         # SSSP precision max(Au 200, Hg 390)
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,          # Ry
        "nspin": 1,               # Hg 6s2 closed shell; Au slab metallic; spin-unpolarized
    },
    "electrons": {
        "conv_thr": 1.0e-8,       # Ry (scaled with system size below for slabs)
        "mixing_beta": 0.3,
        "mixing_mode": "local-TF",
        "electron_maxstep": 200,
    },
    # upscale=1: keep conv_thr fixed during relaxations. With QE's default (100) conv_thr is
    # tightened to 1e-10 Ry near convergence, below the ~2e-9 Ry noise floor of the GPU build
    # (an SCF stalled for 200 iterations, see work/dft/runs/failed/PBE_site_hcp_gpu_scf_stall).
    "ions": {"ion_dynamics": "bfgs", "upscale": 1.0},
    "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.1},
}

FUNCTIONALS = {
    "PBE": {},
    # Grimme D3 with Becke-Johnson damping, three-body ATM term as QE default (on)
    "PBE-D3": {"vdw_corr": "dft-d3", "dftd3_version": 4, "dftd3_threebody": True},
}

ENGINES = {
    # name: (launcher, default MPI ranks, extra pw.x args)
    "cpu": (os.path.join(LAUNCHERS, "pw-mkl.x"), 10, []),     # QE 7.6, MKL, MPI
    "gpu": (os.path.join(LAUNCHERS, "pw-gpu.x"), 1, []),      # QE 7.6, CUDA
}


def build_input(functional, calculation, overrides=None):
    d = json.loads(json.dumps(BASE))
    d["control"]["calculation"] = calculation
    d["system"].update(FUNCTIONALS[functional])
    for sec, vals in (overrides or {}).items():
        d.setdefault(sec, {}).update(vals)
    if calculation not in ("relax", "vc-relax"):
        d.pop("ions")
    if calculation != "vc-relax":
        d.pop("cell")
    return d


def _done(pwo):
    return os.path.exists(pwo) and "JOB DONE" in open(pwo, errors="ignore").read()


def _busy(wdir):
    """True if another worker is running this calculation: a lock file owned by a live process,
    or (for runs started before locks existed) a pw.out without JOB DONE written to in the last 15 min."""
    lock, pwo = os.path.join(wdir, "LOCK"), os.path.join(wdir, "pw.out")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().split()[0])
            os.kill(pid, 0)
            return pid != os.getpid()
        except (ValueError, ProcessLookupError, IndexError):
            os.remove(lock)
    return (os.path.exists(pwo) and not _done(pwo)
            and time.time() - os.path.getmtime(pwo) < 900)


def run_pw(atoms, label, functional, calculation="scf", kpts=None, overrides=None,
           engine="cpu", np_=None, npool=1, reuse=True, on_busy="wait"):
    """Write input, run pw.x, return dict(energy_eV, atoms, wall_s, n_scf, ...).
    If another worker is running the same label: on_busy='wait' polls until it is done and then
    reads its result; on_busy='skip' returns None immediately."""
    wdir = os.path.join(RUN_ROOT, label)
    os.makedirs(wdir, exist_ok=True)
    pwi, pwo = os.path.join(wdir, "pw.in"), os.path.join(wdir, "pw.out")
    while not _done(pwo) and _busy(wdir):
        if on_busy == "skip":
            return None
        time.sleep(30)
    lock = os.path.join(wdir, "LOCK")
    if not (reuse and _done(pwo)):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:     # lost a start race against another worker
            return run_pw(atoms, label, functional, calculation, kpts, overrides, engine, np_,
                          npool, reuse, on_busy)
        os.write(fd, f"{os.getpid()} {engine}\n".encode()); os.close(fd)
    inp = build_input(functional, calculation, overrides)
    inp["control"]["prefix"] = "pw"
    inp["control"]["outdir"] = os.path.join(wdir, "tmp")
    prev = {}
    if not (reuse and _done(pwo)):
        write(pwi, atoms, format="espresso-in", input_data=inp, pseudopotentials=PSEUDOS,
              kpts=kpts, koffset=(0, 0, 0))
        split_fixed_species(pwi)
        launcher, np_def, extra = ENGINES[engine]
        env = dict(os.environ, NP=str(np_ or np_def), OMP_NUM_THREADS="1")
        cmd = [launcher] + extra + (["-nk", str(npool)] if npool > 1 else []) + ["-in", pwi]
        t0 = time.time()
        try:
            with open(pwo, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=wdir)
        finally:
            if os.path.exists(lock):
                os.remove(lock)
        wall = time.time() - t0
    else:
        wall = None
        rj = os.path.join(wdir, "run.json")
        if os.path.exists(rj):      # keep the record of the worker that actually ran it
            prev = {k: v for k, v in json.load(open(rj)).items()
                    if k in ("engine", "np", "npool", "wall_python_s")}
    shutil.rmtree(os.path.join(wdir, "tmp"), ignore_errors=True)
    out = open(pwo, errors="ignore").read()
    res = parse(out)
    res.update(label=label, functional=functional, calculation=calculation,
               kpts=list(kpts) if kpts else "gamma", engine=engine,
               np=np_ or ENGINES[engine][1], npool=npool, wall_python_s=wall,
               ecutwfc=inp["system"]["ecutwfc"], ecutrho=inp["system"]["ecutrho"])
    res.update(prev)
    try:
        res["atoms"] = read(pwo, index=-1, format="espresso-out")
    except Exception as e:      # failed run
        res["atoms"] = None
        res["read_error"] = str(e)
    log = {k: v for k, v in res.items() if k != "atoms"}
    log["input"] = inp
    with open(os.path.join(wdir, "run.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    return res


def split_fixed_species(pwi):
    """Give fully fixed atoms (if_pos 0 0 0) their own species label, e.g. Au -> Au1 (same
    pseudopotential and mass). Otherwise QE may keep symmetry operations (e.g. inversion of the
    4-layer slab) that map fixed onto free atoms; the first BFGS step then breaks them and pw.x
    stops with 'checkallsym: some of the original symmetry operations not satisfied'."""
    lines = open(pwi).read().splitlines()
    i_pos = next(i for i, l in enumerate(lines) if l.startswith("ATOMIC_POSITIONS"))
    nat = int(next(l for l in lines if l.strip().startswith("nat")).split("=")[1])
    new_species = {}
    for j in range(i_pos + 1, i_pos + 1 + nat):
        p = lines[j].split()
        if len(p) == 7 and p[4:] == ["0", "0", "0"]:
            new_species[p[0]] = p[0] + "1"
            lines[j] = lines[j].replace(p[0], p[0] + "1", 1)
    if not new_species:
        return
    i_sp = next(i for i, l in enumerate(lines) if l.startswith("ATOMIC_SPECIES"))
    j = i_sp + 1
    while lines[j].strip():
        p = lines[j].split()
        if p[0] in new_species:
            lines.insert(j + 1, f"{new_species[p[0]]} {p[1]} {p[2]}")
            j += 1
        j += 1
    k = next(i for i, l in enumerate(lines) if l.strip().startswith("ntyp"))
    ntyp = int(lines[k].split("=")[1])
    lines[k] = f"   ntyp             = {ntyp + len(new_species)}"
    open(pwi, "w").write("\n".join(lines) + "\n")


def parse(out):
    lines = out.splitlines()
    e = [float(l.split("=")[1].split()[0]) for l in lines if l.startswith("!")]
    n_iter = [int(l.split("achieved in")[1].split()[0]) for l in lines
              if "convergence has been achieved in" in l]
    wall = [l for l in lines if l.strip().startswith("PWSCF") and "WALL" in l]
    nirr = [int(l.split("=")[1].split()[0]) for l in lines if "number of k points=" in l]
    res = dict(
        energy_eV=e[-1] * 13.605693122994 if e else None,
        n_scf=len(e),
        scf_iterations=n_iter,
        job_done="JOB DONE" in out,
        relax_converged=("bfgs converged" in out) or ("End of BFGS Geometry Optimization" in out
                                                       and "bfgs failed" not in out),
        bfgs_failed="bfgs failed" in out or "The maximum number of steps has been reached" in out,
        scf_not_converged="convergence NOT achieved" in out,
        wall_pw=wall[-1].strip() if wall else None,
        nks_irr=nirr[0] if nirr else None,
    )
    tf = [l for l in lines if "Total force =" in l]
    if tf:
        res["total_force_Ry_bohr"] = float(tf[-1].split("=")[1].split()[0])
    return res
