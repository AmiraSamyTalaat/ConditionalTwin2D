# Cohort computational provenance

This table records only provenance supported by the frozen source package and revised manuscript. Fields not explicitly available are marked as such rather than inferred.

## QPOD/C2DB development
- **records:** 529
- **hosts:** 30
- **study role:** Development; LOHO cross-fitted evaluation and model selection
- **operations:** substitution, vacancy, mixed
- **charge states:** Frozen development identifiers are charge-state 0 rows in the QPOD naming scheme
- **moment endpoint:** Study endpoint uses absolute net magnetic moment of the defective supercell; primary event is >=0.10 μB
- **electronic structure:** GPAW; spin-polarized PBE
- **plane wave or grid cutoff:** 800 eV plane-wave cutoff
- **kpoint sampling:** k-point density 6 Å^-1 for relaxations and 12 Å^-1 for ground-state calculations
- **supercell or defect separation:** Fixed supercells chosen to keep periodic defect images >15 Å apart
- **vacuum or dimensional setting:** 2D source structures; graph audit uses pbc=[True,True,False]
- **relaxation:** Atomic relaxation to forces <0.01 eV Å^-1
- **filtering overlap:** Development manifest derived from QPOD defects paired to C2DB pristine hosts; structure-JSON magnetic moment treated as authoritative when metadata disagreed
- **defect concentration note:** No single uniform concentration range is recorded in the frozen development manifest; do not infer one from atom-count/area features.
- **evidence source:** Revised manuscript Section 2.2; archived source/configuration and development manifests

## 2DMD strict
- **records:** 2500
- **hosts:** 5
- **study role:** Original prediction-sealed same-regime external cohort for DefectTwin; retrospective diagnostic for ConditionalTwin
- **operations:** substitution, vacancy, mixed
- **charge states:** Source dataset reported without charged-state calculations
- **moment endpoint:** Mapped to the study endpoint: absolute net magnetic moment of the defective supercell; primary event >=0.10 μB
- **electronic structure:** VASP/PAW; spin-polarized PBE
- **plane wave or grid cutoff:** 500 eV
- **kpoint sampling:** Gamma-point structural relaxation for large material-specific supercells
- **supercell or defect separation:** Large material-specific supercells; a uniform defect-concentration range is not available in the frozen source package
- **vacuum or dimensional setting:** >=15 Å vacuum
- **relaxation:** Force tolerance 0.01 eV Å^-1
- **filtering overlap:** Strict external cohort retained as supplied for the sealed base evaluation; no external labels used for feature/model selection; only three of five hosts are class-complete at the primary endpoint
- **defect concentration note:** Not uniformly available in the frozen source package; no range is inferred.
- **evidence source:** Revised manuscript Section 2.2; 2DMD frozen feature/evaluated files and per-host audit

## Imp2D strict
- **records:** 11866
- **hosts:** 40
- **study role:** Original prediction-sealed cross-family external cohort for DefectTwin; retrospective diagnostic for ConditionalTwin
- **operations:** adsorbate, interstitial
- **charge states:** Neutral charge states
- **moment endpoint:** Mapped to the study endpoint: absolute net magnetic moment of the defective supercell; primary event >=0.10 μB
- **electronic structure:** VASP/PAW; spin-polarized PBE
- **plane wave or grid cutoff:** 600 eV plane-wave cutoff; 900 eV kinetic cutoff
- **kpoint sampling:** Gamma-point sampling
- **supercell or defect separation:** >=10 Å defect-image separation
- **vacuum or dimensional setting:** 2D source structures; no additional uniform vacuum value is asserted from the frozen package
- **relaxation:** Two-stage relaxation workflow
- **filtering overlap:** Four external phases overlapping the development set were excluded from the strict cohort; row-level phase audit identified 45 unit-cell modes across 44 pre-strict hosts
- **defect concentration note:** No single uniform defect-concentration range is recorded in the frozen strict cohort; no range is inferred.
- **evidence source:** Revised manuscript Sections 2.2 and 2.8; frozen Imp2D files

## QPOD release extension
- **records:** 693
- **hosts:** 7
- **study role:** Precommitted same-repository release extension for DefectTwin; retrospective diagnostic for ConditionalTwin
- **operations:** substitution, vacancy, mixed
- **charge states:** Charge-state field is not explicit in the frozen extension table; record IDs are not used to infer charge state
- **moment endpoint:** Study endpoint uses absolute net magnetic moment of the defective supercell; primary event >=0.10 μB
- **electronic structure:** QPOD source workflow: GPAW; spin-polarized PBE
- **plane wave or grid cutoff:** QPOD source workflow: 800 eV plane-wave cutoff
- **kpoint sampling:** QPOD source workflow: k-point density 6 Å^-1 relaxations and 12 Å^-1 ground state
- **supercell or defect separation:** QPOD source workflow uses fixed supercells chosen for >15 Å periodic defect-image separation
- **vacuum or dimensional setting:** Exact-phase-unseen, formula-overlapping same-repository extension; source is 2D
- **relaxation:** QPOD source workflow: forces <0.01 eV Å^-1
- **filtering overlap:** 693 valid records from seven exact-phase-unseen yet formula-overlapping hosts; retained without recalibration, host exclusion, or threshold modification
- **defect concentration note:** No uniform concentration range is explicitly available in the frozen extension table; no range is inferred.
- **evidence source:** Revised manuscript Sections 2.2 and 3.8; qpod_features.csv/qpod_evaluated.csv
