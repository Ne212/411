# Pseudopotentials (not included)

The pseudopotential files are not redistributed here. Download them from SSSP 1.3.0 PBE precision
(Materials Cloud, https://www.materialscloud.org/discover/sssp) and put them in this folder:

| Element | File | Type | md5 |
|---|---|---|---|
| Au | `Au_ONCV_PBE-1.0.oncvpsp.upf` | SG15 ONCV, norm-conserving, scalar-relativistic | `d14007822ba0e19f0206237467fc06c2` |
| Hg | `Hg.us.z_12.uspp.gbrv.v1.upf` | GBRV 1.0, ultrasoft, scalar-relativistic | `8f42caffe07f7fd9c0a564347f8c5498` |

`work/dft/qe_common.py` reads them from this directory (`PSEUDO_DIR`).
