"""Parse Tables I-III of Steckel, PRB 77, 115412 (2008) from the pdftotext output of the OSTI copy and write
results/published/steckel2008_systems.csv (44 systems). Values are parsed, not typed by hand.
Usage: python3 -I build_steckel_table.py <steckel.txt>"""
import csv, re, sys, os
txt = open(sys.argv[1], encoding="utf-8").read().splitlines()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "published", "steckel2008_systems.csv")
num = r"([−-]?\d+\.\d+)"
row_re = re.compile(r"^\s*(\S.*?-Hg/\s*(\w+)共(111|001)兲)\s+" + r"\s+".join([num] * 7) + r"\s*$")
COMMON = dict(source="Steckel, Phys. Rev. B 77, 115412 (2008)", doi="10.1103/PhysRevB.77.115412",
              code="VASP, PAW, Ecut 380 eV, Methfessel-Paxton 0.05 eV, dipole correction",
              functional="PBE (main text, after Table I: all hollow-site data 'generated using the PBE "
                         "functional'; Table I has PBE/PW91/LDA columns, the PBE column is used)",
              E_ads_definition="'binding energy' of Hg at hollow site vs isolated gas-phase Hg atom; explicit "
                               "equation and per-Hg normalisation not given in main text (not verified)",
              slab_model="asymmetric 4-layer slab, bottom 2 layers fixed at bulk, top 2 + Hg relaxed (forces < 1e-5 eV/A), "
                         "vacuum >= ~10 A, PBE bulk lattice constant (values in suppl., not available)",
              spin="not stated in main text (not verified)")
cells = {"共2冑3 ⫻ 3兲": ("(2sqrt3 x 3)", 1), "p共2 ⫻ 2兲": ("p(2x2)", None), "共冑3 ⫻ 冑3兲R30°": ("(sqrt3 x sqrt3)R30", 1),
         "c共4 ⫻ 4兲": ("c(4x4)", 2), "c共2 ⫻ 2兲": ("c(2x2)", 1), "c共1 ⫻ 1兲": ("c(1x1)", 1)}
rows = []
for ln, line in enumerate(txt, 1):
    m = row_re.match(line)
    if not m:
        continue
    label, metal, facet = m.group(1), m.group(2), m.group(3)
    vals = [float(v.replace("−", "-")) for v in m.groups()[3:]]
    theta, eads, dmhg, dhghg = vals[:4]
    note = ""
    if metal == "Ci":
        metal, note = "Cu", "printed 'Ci(111)' in Table II, a typo for Cu (row order Ag, Au, Cu, Ni, Pd, Pt)"
    cell = next(v for k, v in cells.items() if label.replace(" ", "").startswith(k.replace(" ", "")))
    n_hg = cell[1] if cell[1] is not None else (2 if abs(theta - 0.5) < 1e-6 else 1)
    site = "hollow (fcc/hcp: 'negligible' difference per paper)" if facet == "111" else "4-fold hollow"
    if facet == "111" and n_hg == 2:
        site = "hollow, 2 Hg per p(2x2): honeycomb (no Hg at hexagon centre), i.e. fcc + hcp (inferred from text/Fig. 1)"
    rows.append(dict(id=f"S08-{metal}{facet}-{theta:g}", table="II" if facet == "111" else "III", txt_line=ln,
                     metal=metal, facet=f"({facet})", overlayer=cell[0], coverage_ML=theta, n_Hg_per_cell=n_hg,
                     site=site, E_ads_eV=eads, d_M_Hg_A=dmhg, d_Hg_Hg_A=dhghg, note=note, **COMMON))
# Table I: Au(001) c(4x4) bridge and top (hollow duplicates Table III), PBE column
for ln, line in enumerate(txt, 1):
    # two-column layout: Table I rows share the line with left-column body text -> match at line end
    m = re.search(r"\b(Bridge|Top)\s+" + r"\s+".join([num] * 4) + r"\s*$", line)
    if m:
        rows.append(dict(id=f"S08-Au001-0.125-{m.group(1).lower()}", table="I", txt_line=ln, metal="Au", facet="(001)",
                         overlayer="c(4x4)", coverage_ML=0.125, n_Hg_per_cell=2, site=m.group(1).lower(),
                         E_ads_eV=float(m.group(2).replace("−", "-")), d_M_Hg_A=None, d_Hg_Hg_A=None,
                         note="Table I, PBE column; M-Hg distance not given", **COMMON))
# per-Hg vs per-cell normalisation is NOT stated anywhere in the paper (methods, Table I-III captions).
# The text only says that the 0.25 ML and 0.5 ML p(2x2)/(111) overlayers use the same supercell with one and two
# Hg atoms. The number of Hg in the c(4x4) supercell is not stated. Rows whose supercell has (or may have) more
# than one Hg are therefore excluded from the main MAE and reported separately.
for r in rows:
    multi = (r["overlayer"] == "p(2x2)" and abs(r["coverage_ML"] - 0.5) < 1e-6) or r["overlayer"] == "c(4x4)"
    r["normalisation"] = ("unverified: >1 Hg per supercell (0.5 ML p(2x2) has 2 Hg per text; c(4x4) Hg count not "
                          "stated), per-Hg vs per-cell not stated") if multi else "1 Hg per cell: normalisation irrelevant"
    r["in_main_MAE"] = not multi
    r["Ni_flag"] = ("Ni is ferromagnetic; spin treatment in the paper not stated, MLIPs have no spin input"
                    if r["metal"] == "Ni" else "")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "systems ->", OUT)
