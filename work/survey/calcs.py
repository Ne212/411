"""
Calculator registry for the MLIP survey.

Every model is built here and only here, so all scripts (smoke test, Hg/Au(111) protocol,
benchmarks) use identical settings.  Each entry returns (ase_calculator, info_dict).
Imports are lazy: a model can only be built inside its own venv (see VENV below).

Usage:  from calcs import make_calc, VENV;  calc, info = make_calc("MACE-MP-0-medium")
"""
import os

DEVICE = os.environ.get("SURVEY_DEVICE", "cuda")

# model key -> venv name under START/venvs
VENV = {
    "MACE-MP-0-medium": "mace", "MACE-MPA-0-medium": "mace", "MACE-OMAT-0-medium": "mace",
    "MACE-MH-1-omat": "mace", "MACE-MH-1-oc20": "mace",
    "CHGNet-0.3.0": "chgnet",
    "TensorNet-MatPES-PBE": "matgl", "M3GNet-MatPES-PBE": "matgl", "CHGNet-MatPES-PBE": "matgl",
    "ALIGNN-FF-mptrj": "alignn", "ALIGNN-FF-omat24": "alignn", "ALIGNN-FF-matpes-pbe": "alignn",
    "SevenNet-Omni-mpa": "sevenn", "SevenNet-Omni-omat24": "sevenn", "SevenNet-Omni-oc20": "sevenn",
    "SevenNet-MF-ompa-mpa": "sevenn",
    "MatterSim-5M": "mattersim",
    "ORB-v3-cons-inf-omat": "orb", "ORB-v3-cons-inf-mpa": "orb",
    "UMA-s-1p2p1-omat": "fairchem", "UMA-s-1p2p1-oc20": "fairchem",
    "GRACE-2L-OAM-L": "grace", "GRACE-2L-SMAX-OMAT-L": "grace",
    "NequIP-OAM-L": "nequip", "Allegro-OAM-L": "nequip",
    "DPA-3.1-3M-omat24": "deepmd", "DPA-3.1-3M-mptrj": "deepmd", "DPA-3.1-3M-oc20m": "deepmd",
    "DPA-3.1-3M-AgAu-PBED3": "deepmd", "DPA4-Plus-OMat24": "deepmd",
    "PET-OAM-L": "upet",
    # Stage 4 fine-tuned models (work/survey/finetune)
    "MACE-MP-0-ft-noatom": "mace", "MACE-MP-0-ft-atom": "mace",
    "CHGNet-ft-noatom": "chgnet", "CHGNet-ft-atom": "chgnet",
    "MACE-MP-0-ft-atom-replay": "mace", "MACE-MP-0-ft-atom-lr1e-4": "mace",
}
FT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune", "runs")


def _ver(mod):
    import importlib.metadata as md
    try:
        return md.version(mod)
    except Exception:
        return "?"


def _fix_force_shape(calc):
    """ALIGNN 2026.11.1 returns forces of shape (3,) for a 1-atom cell (squeezed batch).
    Reshape to (natoms, 3) after every calculate(); values are untouched."""
    orig = calc.calculate

    def calculate(atoms=None, properties=("energy",), system_changes=None, **kw):
        if system_changes is None:
            orig(atoms, properties, **kw)
        else:
            orig(atoms, properties, system_changes, **kw)
        if "forces" in calc.results:
            calc.results["forces"] = calc.results["forces"].reshape(-1, 3)
    calc.calculate = calculate
    return calc


class FullPBC:
    """For calculators that reject mixed pbc (ORB, PET): evaluate a copy with pbc=True.
    Only allowed if the vacuum gap along each non-periodic axis is >= min_gap (default 7 A: larger
    than the 6 A edge cutoff of ORB and the 4.5 A of PET, so no graph edge can cross the vacuum and
    periodic images cannot interact). The first version used 12 A, which rejected the last point
    (Hg 12 A above the slab, gap 11.99 A) of the dissociation scan."""

    def __new__(cls, inner, min_gap=7.0, full_pbc=True):
        from ase.calculators.calculator import Calculator, all_changes
        import numpy as np

        class _W(Calculator):
            implemented_properties = ["energy", "free_energy", "forces", "stress"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                a = self.atoms.copy()
                for ax in (np.where(~a.pbc)[0] if full_pbc else []):
                    z = a.positions[:, ax]
                    gap = a.cell[ax, ax] - (z.max() - z.min())
                    if gap < min_gap:
                        raise ValueError(f"vacuum gap {gap:.2f} A < {min_gap} A along axis {ax}")
                if full_pbc:
                    a.pbc = True
                a.calc = inner
                self.results["energy"] = self.results["free_energy"] = a.get_potential_energy()
                self.results["forces"] = a.get_forces()
                if "stress" in properties:
                    self.results["stress"] = a.get_stress()
        return _W()


def make_calc(key, device=None):
    dev = device or DEVICE
    k = key

    # ------------------------------------------------------------------ Stage 4 fine-tuned models
    if k.startswith("MACE-MP-0-ft-"):
        from mace.calculators import MACECalculator
        path = os.path.join(FT_DIR, k, f"{k}.model")
        kw = dict(head="Default") if k.endswith("-replay") else {}       # multihead model: our fine-tuned head
        calc = MACECalculator(model_paths=path, device=dev, default_dtype="float64", **kw)
        return calc, dict(model=k, base="MACE-MP-0 medium", checkpoint=path, package=f"mace-torch {_ver('mace-torch')}",
                          device=dev, dtype="float64", dispersion=False, head=kw.get("head"))
    if k.startswith("CHGNet-ft-"):
        import glob
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        path = sorted(glob.glob(os.path.join(FT_DIR, k, "bestE_*.pth.tar")))[-1]
        calc = CHGNetCalculator(model=CHGNet.from_file(path), use_device=dev, check_cuda_mem=False)
        return calc, dict(model=k, base="CHGNet 0.3.0", checkpoint=path, package=f"chgnet {_ver('chgnet')}",
                          device=dev, dtype="float32")

    # ------------------------------------------------------------------ MACE
    if k.startswith("MACE-"):
        from mace.calculators import mace_mp
        name, head = {
            "MACE-MP-0-medium": ("medium", None),
            "MACE-MPA-0-medium": ("medium-mpa-0", None),
            "MACE-OMAT-0-medium": ("medium-omat-0", None),
            "MACE-MH-1-omat": ("mh-1", "omat_pbe"),
            "MACE-MH-1-oc20": ("mh-1", "oc20_usemppbe"),
        }[k]
        kw = dict(model=name, device=dev, default_dtype="float64", dispersion=False)
        if head:
            kw["head"] = head
        calc = mace_mp(**kw)
        return calc, dict(model=k, mace_name=name, head=head, package=f"mace-torch {_ver('mace-torch')}",
                          device=dev, dtype="float64", dispersion=False)

    # ------------------------------------------------------------------ CHGNet (original package)
    if k == "CHGNet-0.3.0":
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator(use_device=dev, check_cuda_mem=False)
        return calc, dict(model=k, package=f"chgnet {_ver('chgnet')}", device=dev, dtype="float32")

    # ------------------------------------------------------------------ MatGL
    if k in ("TensorNet-MatPES-PBE", "M3GNet-MatPES-PBE", "CHGNet-MatPES-PBE"):
        import matgl
        from matgl.ext.ase import PESCalculator
        # matgl 4.1.4: PESCalculator has no device argument and torch.set_default_device("cuda")
        # gives a cuda/cpu mismatch inside the graph converter -> MatGL runs on the CPU.
        dev = "cpu"
        hf = {"TensorNet-MatPES-PBE": "TensorNet-PES-MatPES-PBE-2025.2",
              "M3GNet-MatPES-PBE": "M3GNet-PES-MatPES-PBE-2025.2",
              "CHGNet-MatPES-PBE": "CHGNet-PES-MatPES-PBE-2025.2.10"}[k]
        pot = matgl.load_model(hf)
        calc = PESCalculator(pot, stress_unit="eV/A3")   # default is GPa, ASE needs eV/A^3
        return calc, dict(model=k, hf_repo=f"materialyze/{hf}", package=f"matgl {_ver('matgl')}",
                          device=dev, dtype="float32", stress_unit="eV/A3")

    # ------------------------------------------------------------------ ALIGNN-FF
    if k.startswith("ALIGNN-FF"):
        from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff
        name = {"ALIGNN-FF-mptrj": "alignn_ff_mptrj", "ALIGNN-FF-omat24": "alignn_ff_omat24_1m",
                "ALIGNN-FF-matpes-pbe": "alignn_ff_matpes_pbe"}[k]
        path = get_figshare_model_ff(model_name=name)
        calc = _fix_force_shape(AlignnAtomwiseCalculator(path=path, device=dev))
        return calc, dict(model=k, alignn_name=name, patch="forces reshaped to (N,3) for 1-atom cells", package=f"alignn {_ver('alignn')}", device=dev,
                          dtype="float32")

    # ------------------------------------------------------------------ SevenNet
    if k.startswith("SevenNet"):
        from sevenn.calculator import SevenNetCalculator
        mdl, modal = {"SevenNet-Omni-mpa": ("7net-omni", "mpa"),
                      "SevenNet-Omni-omat24": ("7net-omni", "omat24"),
                      "SevenNet-Omni-oc20": ("7net-omni", "oc20"),
                      "SevenNet-MF-ompa-mpa": ("7net-mf-ompa", "mpa")}[k]
        calc = SevenNetCalculator(model=mdl, modal=modal, device=dev)
        return calc, dict(model=k, sevenn_model=mdl, modal=modal, package=f"sevenn {_ver('sevenn')}",
                          device=dev, dtype="float32")

    # ------------------------------------------------------------------ MatterSim
    if k == "MatterSim-5M":
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=dev)
        return calc, dict(model=k, checkpoint="MatterSim-v1.0.0-5M", package=f"mattersim {_ver('mattersim')}",
                          device=dev, dtype="float32")

    # ------------------------------------------------------------------ ORB
    if k.startswith("ORB-"):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        fn = {"ORB-v3-cons-inf-omat": pretrained.orb_v3_conservative_inf_omat,
              "ORB-v3-cons-inf-mpa": pretrained.orb_v3_conservative_inf_mpa}[k]
        # compile=False: torch.compile needs Python.h, absent for the system python3.12 used by this venv
        orbff, adapter = fn(device=dev, precision="float32-highest", compile=False)
        calc = FullPBC(ORBCalculator(orbff, atoms_adapter=adapter, device=dev))
        return calc, dict(model=k, package=f"orb-models {_ver('orb-models')}", device=dev,
                          wrapper="FullPBC: slabs evaluated with pbc=True (vacuum gap >= 12 A checked)",
                          precision="float32-highest", compile=False)

    # ------------------------------------------------------------------ FAIRChem UMA
    if k.startswith("UMA-"):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        task = k.rsplit("-", 1)[1]
        # "batch" preset: full FP32 (tf32=False), no torch.compile, unmerged MOLE. The default preset
        # ("nitro") uses TF32 matmuls and needs torch.compile/triton, which fails here (no Python.h for
        # the system python3.12).
        pred = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device=dev, inference_settings="batch")
        calc = FAIRChemCalculator(pred, task_name=task)
        return calc, dict(model=k, checkpoint="uma-s-1p2p1", task=task, inference_settings="batch (fp32, no compile)",
                          package=f"fairchem-core {_ver('fairchem-core')}", device=dev)

    # ------------------------------------------------------------------ GRACE
    if k.startswith("GRACE-"):
        from tensorpotential.calculator.foundation_models import grace_fm
        name = {"GRACE-2L-OAM-L": "GRACE-2L-OMAT-large-ft-AM-fp64",
                "GRACE-2L-SMAX-OMAT-L": "GRACE-2L-SMAX-OMAT-large-fp64"}[k]
        calc = grace_fm(name)
        return calc, dict(model=k, grace_name=name, package=f"tensorpotential {_ver('tensorpotential')}",
                          device="TF default (GPU if visible)", dtype="float64")

    # ------------------------------------------------------------------ NequIP / Allegro
    if k in ("NequIP-OAM-L", "Allegro-OAM-L"):
        from nequip.integrations.ase import NequIPCalculator
        path = os.path.join(os.path.dirname(__file__), "models", f"{k}.nequip.pt2")
        calc = NequIPCalculator.from_compiled_model(path, device=dev)
        return calc, dict(model=k, nequip_id=f"nequip.net:mir-group/{k}:0.1", compiled=path,
                          package=f"nequip {_ver('nequip')} / nequip-allegro {_ver('nequip-allegro')}",
                          device=dev)

    # ------------------------------------------------------------------ DeePMD DPA-3 / DPA-4
    if k.startswith("DPA"):
        from deepmd.calculator import DP
        cache = os.path.expanduser("~/.cache/deepmd/pretrained/models")
        fname, head = {"DPA-3.1-3M-omat24": ("DPA-3.1-3M", "Omat24"),
                       "DPA-3.1-3M-mptrj": ("DPA-3.1-3M", "MP_traj_v024_alldata_mixu"),
                       "DPA-3.1-3M-oc20m": ("DPA-3.1-3M", "OC20M"),
                       "DPA-3.1-3M-AgAu-PBED3": ("DPA-3.1-3M", "Metals_AgAu_PBED3"),
                       "DPA4-Plus-OMat24": ("DPA4-Plus-OMat24-v20260805", None)}[k]
        kw = dict(head=head) if head else {}
        calc = DP(model=os.path.join(cache, fname + ".pt"), **kw)
        return calc, dict(model=k, checkpoint=fname, head=head, package=f"deepmd-kit {_ver('deepmd-kit')}",
                          device="torch default (cuda)")

    # ------------------------------------------------------------------ UPET (PET-OAM)
    if k == "PET-OAM-L":
        from upet.ase import UPETCalculator
        # FullPBC: UPETCalculator (a) builds no neighbour list for mixed pbc slabs ("No neighbor-list
        # method fits the current safety and feasibility guards") and (b) does not provide
        # 'free_energy', which ASE's numerical-stress helper requests; free_energy := energy.
        calc = FullPBC(UPETCalculator(model="pet-oam-l", version="0.1.0", device=dev))
        return calc, dict(model=k, upet_model="pet-oam-l", version="0.1.0", package=f"upet {_ver('upet')}",
                          device=dev, wrapper="FullPBC (vacuum gap >= 12 A checked); free_energy := energy")

    raise KeyError(k)
