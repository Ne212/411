# Experimental adsorption enthalpy of Hg on gold: verified sources

All DOIs were checked against Crossref (api.crossref.org) on 2026-10-29. "Verified in text"
means I read the number in a full text myself. "Not accessible" means the primary paper
was paywalled or blocked, so the value comes second-hand from a citing paper (stated per entry).

| Quantity | Value | Method | Source | Status |
|---|---|---|---|---|
| −ΔH_ads^Au(Hg) | **98 ± 3 kJ/mol** (1.02 ± 0.03 eV) | off-line gas thermochromatography, Au surface (polycrystalline), zero-coverage, single-atom regime | Soverna S. et al., *Radiochim. Acta* **93**, 1–8 (2005), doi:10.1524/ract.93.1.1.58298 | Primary full text not accessible (403/405). The value, with ±3 and attribution to Soverna 2005, is verified in the text of Lens et al. 2018 (next row, their ref. 31). Eichler 2013 (row 3) uses −98 kJ/mol in its Monte-Carlo simulations of the Hg distribution. |
| −ΔH_ads^Au(Hg) | **> 67 kJ/mol** (lower limit, 95 % c.i.) | on-line gas chromatography at TASCA (GSI), ^182–184Hg on Au-covered detector array (COMPACT II), Monte-Carlo simulation | Lens L. et al., *Radiochim. Acta* **106**, 949–962 (2018), doi:10.1515/ract-2017-2914 | Verified in text (LLNL-JRNL-838949 manuscript, OSTI 1898301). The authors state that MCS with 98 ± 3 kJ/mol [Soverna 2005] "describes our experimental observation well". |
| −ΔH_ads^Au(Hg) | 98 kJ/mol (used in simulation) | thermochromatography, Cn experiments at FLNR Dubna | Eichler R., *J. Phys.: Conf. Ser.* **420**, 012003 (2013), doi:10.1088/1742-6596/420/1/012003 (arXiv:1212.4292) | Verified in text (Fig. 5 caption). The Hg point in its Fig. 6 cites B. Eichler, *Kernenergie* 19, 307 (1976) and PSI Report 00-09 (2000). |
| −ΔH_ads^Au(Cn) | 52 (+3, −2) kJ/mol | thermochromatography of ^283Cn on Au | as quoted in Eichler 2013 (above); originally Eichler R. et al., *Nature* **447**, 72–75 (2007), doi:10.1038/nature05761 and *Angew. Chem. Int. Ed.* **47**, 3262–3266 (2008), doi:10.1002/anie.200705019 | Value verified in the Eichler 2013 text. The Nature/Angew primary texts were not read. |
| −ΔH_ads^Au(Rn) | 20 kJ/mol (simulation value) | — | Eichler 2013 (Fig. 5 caption) | Verified in text |
| E_coh(Hg) | 0.67 eV/atom (65 kJ/mol) | thermochemical | C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley (2005), Ch. 3, Table 1 | From memory of a standard textbook table. **Not re-checked against the book in this session**; flag before final submission. |

## Notes for the comparison with DFT
* ΔH_ads from thermochromatography is a zero-coverage enthalpy on polycrystalline Au (not a
  well-defined Au(111) facet), derived through a mobile-adsorption model. Comparing it with a
  0 K electronic E_ads on Au(111) at 1/9 ML neglects ZPE/thermal corrections (expected to be a
  few kJ/mol for a heavy atom), surface heterogeneity (steps and defects bind more strongly) and
  model assumptions in the MC analysis.
* 1 eV = 96.485 kJ/mol, so 98 kJ/mol = 1.016 eV and 67 kJ/mol = 0.694 eV.
* I could **not** find a single-crystal surface-science (TPD) value for Hg/Au(111) in this
  session. If one is needed, it has to come from a targeted library search.
