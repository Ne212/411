"""
Stage 3: Hg on metal surfaces vs Steckel, PRB 77, 115412 (2008) (VASP-PBE), set 1 = 44 systems
(results/published/steckel2008_systems.csv).

For every model and every system:
  * the metal's lattice constant is the model's own a0 (Brent on E(a) + Birch-Murnaghan, as in project 411);
  * slab as in the paper: 4 layers, bottom 2 fixed, vacuum 12 A on each side (411 convention; paper: >= ~10 A);
  * Hg in the hollow (111: fcc hollow = above a 3rd-layer atom; 0.5 ML p(2x2): fcc + hcp honeycomb, Hg-Hg = 2 a_s/sqrt3;
    001: 4-fold hollow = above a 2nd-layer atom); Au(001) bridge/top from Table I;
  * mode "source": ideal slab, Hg at the paper's M-Hg distance, single point (clean slab ideal too);
    not possible for bridge/top (no distance given);
  * mode "relaxed": clean slab relaxed (top 2 layers), slab+Hg relaxed (top 2 layers + Hg along z only, so the
    site is kept), BFGS fmax 0.02 eV/A, <= 1000 steps (411 settings);
  * E_ads per Hg, scheme A: [E(slab+nHg) - E(slab) - n E(Hg atom, 15 A box)] / n
    scheme D: E(Hg atom) replaced by E(bulk alpha-Hg, model, from results/hg_au/<model>/bulk.csv) + E_coh,PBE(QE 411).
Our cells may hold more Hg than the paper's (e.g. c(1x1) built as 2x2); energies are per Hg, which equals the
paper's quantity only if the paper is per Hg (verified only for 1-Hg cells, see in_main_MAE in the CSV).
Usage: <venv>/bin/python published_benchmark.py MODEL_KEY  -> results/published/models/<key>.csv
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from ase import Atoms
from ase.build import bulk, fcc100, fcc111
from ase.constraints import FixAtoms, FixedLine
from ase.eos import EquationOfState
from ase.geometry import find_mic
from ase.io import write
from ase.optimize import BFGS

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from calcs import make_calc
RES = os.path.join(HERE, "..", "..", "results")
SET = dict(layers=4, vacuum=12.0, fmax=0.02, max_steps=1000, atom_box=15.0, start_height_top_bridge=2.6,
           eos=dict(points=11, rel_range=0.03), a_guess=dict(Ag=4.09, Au=4.08, Cu=3.61, Ni=3.52, Pd=3.89, Pt=3.92))


def lattice(calc, el):
    g = SET["a_guess"][el]
    f = lambda a: (lambda at: (setattr(at, "calc", calc), at.get_potential_energy())[1])(bulk(el, "fcc", a=a))
    r = minimize_scalar(f, bounds=(0.93 * g, 1.08 * g), method="bounded", options=dict(xatol=1e-5))
    grid = r.x * np.linspace(1 - SET["eos"]["rel_range"], 1 + SET["eos"]["rel_range"], SET["eos"]["points"])
    v, e = [], []
    for a in grid:
        at = bulk(el, "fcc", a=a); at.calc = calc; v.append(at.get_volume()); e.append(at.get_potential_energy())
    # A model that has lost the metal (Stage 4 forgetting test) can give a flat E(a): no minimum inside the
    # bracket or a failed EOS fit -> None, and the systems of that metal are skipped (recorded as failed).
    if np.ptp(e) < 1e-3 or min(abs(r.x - 0.93 * g), abs(r.x - 1.08 * g)) < 1e-3 * g:
        return None
    try:
        v0, e0, B = EquationOfState(v, e, eos="birchmurnaghan").fit()
    except Exception:
        return None
    return float((4 * v0) ** (1 / 3))


def clean_slab(row, a0):
    el, fac, ov = row.metal, row.facet, row.overlayer
    L, vac = SET["layers"], SET["vacuum"]
    if fac == "(111)":
        if ov == "(2sqrt3 x 3)":
            s = fcc111(el, (3, 4, L), a=a0, vacuum=vac, orthogonal=True)
        elif ov == "p(2x2)":
            s = fcc111(el, (2, 2, L), a=a0, vacuum=vac)
        elif ov == "(sqrt3 x sqrt3)R30":
            from ase.build import fcc111_root
            s = fcc111_root(el, 3, (1, 1, L), a=a0, vacuum=vac)
    else:
        n = {"c(4x4)": 4, "c(2x2)": 2, "c(1x1)": 2}[ov]
        s = fcc100(el, (n, n, L), a=a0, vacuum=vac)
    s.set_constraint(FixAtoms(mask=s.get_tags() >= L - 1))
    return s


def hg_xy(s, row):
    """Lateral Hg positions in our cell. Returns (list of xy, kind)."""
    t, P = s.get_tags(), s.positions
    l1, l2, l3 = P[t == 1, :2], P[t == 2, :2], P[t == 3, :2]
    a_s = float(np.min([np.linalg.norm(l1[0] - p) for p in l1[1:]])) if len(l1) > 1 else None
    if row.facet == "(111)":
        fcc = l3[0]
        if row.overlayer == "p(2x2)" and row.n_Hg_per_cell == 2:
            a_s = np.linalg.norm(s.cell[0]) / 2
            d = np.c_[l2 - fcc, np.zeros(len(l2))]
            dm, dist = find_mic(d, s.cell, pbc=[True, True, False])
            j = int(np.argmin(np.abs(dist - 2 * a_s / np.sqrt(3))))
            return [fcc, fcc + dm[j, :2]], "fcc+hcp honeycomb"
        return [fcc], "fcc hollow"
    # (001): 4-fold hollow above a 2nd-layer atom
    h0 = l2[0]; a_s = np.linalg.norm(s.cell[0]) / {"c(4x4)": 4, "c(2x2)": 2, "c(1x1)": 2}[row.overlayer]
    ex, ey = s.cell[0, :2] / np.linalg.norm(s.cell[0, :2]), s.cell[1, :2] / np.linalg.norm(s.cell[1, :2])
    if row.overlayer == "c(1x1)":
        return [h0 + i * a_s * ex + j * a_s * ey for i in (0, 1) for j in (0, 1)], "4-fold hollow"
    if row.overlayer == "c(2x2)":
        return [h0, h0 + a_s * (ex + ey)], "4-fold hollow"
    base = l1[0] if "top" in row.id else (l1[0] + a_s * ex / 2 if "bridge" in row.id else h0)
    kind = "top" if "top" in row.id else ("bridge" if "bridge" in row.id else "4-fold hollow")
    return [base, base + 2 * a_s * (ex + ey)], kind                         # c(4x4): centred, Hg-Hg = 2 sqrt2 a_s


def place_hg(s, xys, d_target):
    """Add Hg above each xy; height chosen so that the nearest-surface-atom distance = d_target (source mode)
    or a start height (relaxed mode, d_target=None)."""
    t = s.get_tags(); top = s.positions[t == 1]; ztop = top[:, 2].mean()
    a = s.copy()
    for xy in xys:
        d = np.c_[top[:, :2] - xy, np.zeros(len(top))]
        _, lat = find_mic(d, s.cell, pbc=[True, True, False])
        r = float(lat.min())
        if d_target is not None:
            if d_target <= r:
                raise ValueError("M-Hg distance smaller than lateral distance")
            h = np.sqrt(d_target ** 2 - r ** 2)
        else:
            h = SET["start_height_top_bridge"] if r < 0.5 else 2.4
        a += Atoms("Hg", positions=[(xy[0], xy[1], ztop + h)])
    return a


def relax(at, calc, label, sdir):
    at.calc = calc
    opt = BFGS(at, logfile=os.path.join(sdir, f"{label}.bfgs.log"))
    conv = bool(opt.run(fmax=SET["fmax"], steps=SET["max_steps"]))
    write(os.path.join(sdir, f"{label}.xyz"), at)
    return at.get_potential_energy(), conv, opt.nsteps


def dmin_hg(at):
    hg = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "Hg"]
    m = [i for i, s in enumerate(at.get_chemical_symbols()) if s != "Hg"]
    return float(np.mean([at.get_distances(i, m, mic=True).min() for i in hg]))


def main(key):
    t0 = time.time()
    sysdf = pd.read_csv(os.path.join(RES, "published", "steckel2008_systems.csv"))
    out = os.path.join(RES, "published", "models"); sdir = os.path.join(out, key); os.makedirs(sdir, exist_ok=True)
    calc, info = make_calc(key)
    hgb = pd.read_csv(os.path.join(RES, "hg_au", key, "bulk.csv")).iloc[0]
    ecoh_pbe = float(pd.read_csv(os.path.join(RES, "dft_bulk_refs.csv")).set_index("functional").loc["PBE", "Ecoh_Hg"])
    hg = Atoms("Hg", cell=[SET["atom_box"]] * 3, pbc=True); hg.center(); hg.calc = calc
    e_atom = hg.get_potential_energy(); e_ref_D = float(hgb.E_Hg_bulk_per_atom) + ecoh_pbe
    a0 = {el: lattice(calc, el) for el in sorted(sysdf.metal.unique())}
    clean_cache, rows = {}, []
    failed = sorted(el for el, v in a0.items() if v is None)
    for _, r in sysdf.iterrows():
        if a0[r.metal] is None:
            rows.append(dict(model=key, system=r.id, metal=r.metal, facet=r.facet, overlayer=r.overlayer,
                             coverage_ML=r.coverage_ML, mode="failed: no bound fcc lattice (flat or minimum-less E(a))",
                             E_ads_ref_eV=r.E_ads_eV, in_main_MAE=bool(r.in_main_MAE),
                             Ni_flag=bool(isinstance(r.Ni_flag, str) and r.Ni_flag)))
            continue
        ck = (r.metal, r.facet, r.overlayer)
        if ck not in clean_cache:
            s = clean_slab(r, a0[r.metal]); s.calc = calc
            e_ideal = s.get_potential_energy()
            sr = s.copy(); e_rel, conv_c, _ = relax(sr, calc, f"clean_{r.metal}{r.facet}_{r.overlayer}".replace(" ", ""), sdir)
            clean_cache[ck] = (s, e_ideal, sr, e_rel, conv_c)
        s, e_ideal, sr, e_rel, conv_c = clean_cache[ck]
        xys, kind = hg_xy(s, r); n = len(xys)
        base = dict(model=key, system=r.id, metal=r.metal, facet=r.facet, overlayer=r.overlayer,
                    coverage_ML=r.coverage_ML, n_Hg_our_cell=n, site_kind=kind, a0_metal_A=a0[r.metal],
                    E_ads_ref_eV=r.E_ads_eV, d_MHg_ref_A=r.d_M_Hg_A, in_main_MAE=bool(r.in_main_MAE),
                    Ni_flag=bool(isinstance(r.Ni_flag, str) and r.Ni_flag))
        # source geometry (single point)
        if not np.isnan(r.d_M_Hg_A):
            a = place_hg(s, xys, float(r.d_M_Hg_A)); a.calc = calc
            e = a.get_potential_energy()
            rows.append(dict(base, mode="source", E_ads_A_eV=(e - e_ideal - n * e_atom) / n,
                             E_ads_D_eV=(e - e_ideal - n * e_ref_D) / n, d_MHg_A=dmin_hg(a), converged=True, n_steps=0))
        # relaxed geometry (start from the source height when known)
        a = place_hg(sr, xys, None if np.isnan(r.d_M_Hg_A) else float(r.d_M_Hg_A))
        nsl = len(sr); cons = [FixAtoms(mask=np.r_[sr.get_tags() >= SET["layers"] - 1, np.zeros(n, bool)])]
        cons += [FixedLine(nsl + i, [0, 0, 1]) for i in range(n)]
        a.set_constraint(cons)
        e, conv, nst = relax(a, calc, f"{r.id}", sdir)
        rows.append(dict(base, mode="relaxed", E_ads_A_eV=(e - e_rel - n * e_atom) / n,
                         E_ads_D_eV=(e - e_rel - n * e_ref_D) / n, d_MHg_A=dmin_hg(a), converged=conv and conv_c,
                         n_steps=nst))
        print(f"{r.id:24s} n={n} {kind:20s} " + " ".join(f"{x['mode']}:{x['E_ads_A_eV']:+.3f}" for x in rows[-2:]
                                                          if x['system'] == r.id) + f" ref {r.E_ads_eV:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{key}.csv"), index=False, float_format="%.5f")
    json.dump(dict(settings=SET, info=info, a0=a0, metals_without_bound_lattice=failed, E_Hg_atom=e_atom, E_ref_D=e_ref_D, Ecoh_PBE=ecoh_pbe,
                   wall_s=time.time() - t0), open(os.path.join(sdir, "settings.json"), "w"), indent=1, default=str)
    print(f"done {key} {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
