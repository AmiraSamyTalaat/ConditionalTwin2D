# ConditionalTwin-2D reproducibility package v8

Manuscript: **Host Conditional Machine Learning Supports Candidate Ranking of Magnetic Point Defects Within Two Dimensional Materials Using Computed Magnetic Moments**

This is the single canonical public-repository package prepared in response to Reviewer 2. It consolidates the corrected DefectTwin-2D base analysis, ConditionalTwin-2D, upstream graph-descriptor provenance, frozen inputs/predictions, and all revision audits. The manuscript itself retains the original submitted figures/tables and adds the reviewer-requested audits directly to the main text; no separate journal Supplementary Information file is required.

## Evidence status
- DefectTwin-2D external Imp2D/2DMD predictions were prediction-sealed before target access.
- ConditionalTwin-2D was designed after those external outcomes were opened; its cross-cohort comparisons are retrospective.
- The 30 host-row-exclusion tables are sensitivity checks that re-score fixed leave-one-host-out cross-fitted component predictions. They are **not** 30 independent replications and **not** nested outer-fold model-selection estimates.
- The separate probability-calibration analysis uses an outer held-out-host loop with inner 3-fold host-grouped calibration.

## Install
```bash
python -m pip install -r requirements.txt
```
The exact tested Python/package environment is also recorded in `environment.yml` and `requirements-lock.txt`.

## Verification
```bash
python verify_repository.py
```
The standard verifier reruns the fast/medium reviewer-facing analyses inside an isolated temporary copy and validates the archived outputs of the computationally expensive mechanism, pool-size, and scale-ablation jobs. It never modifies the deposited archive or its SHA-256 manifest. To rerun every reviewer audit from scratch, use `python verify_repository.py --full-recompute` (computationally expensive).
For a faster integrity/frozen-package check:
```bash
python verify_repository.py --quick
```

## Key directories
- `base_defecttwin_final_q1/` - corrected DefectTwin base analysis and frozen checkpoints.
- `base_project/` - ConditionalTwin-2D source, frozen inputs, predictions, and archived outputs.
- `revision_scripts/` - reviewer-specific audit scripts.
- `revision_results/` - reviewer-specific audit outputs.
- `upstream_descriptor_generation/` - recovered Stage 2A/Stage 2B evidence for graph descriptor construction.
- `COHORT_COMPUTATIONAL_PROVENANCE.csv/.md` - explicit cohort/source computational provenance.
- `MAIN_ITEM_PROVENANCE_MAP.csv/.md` - explicit per-display-item model/split/threshold/seed/calibration/artifact provenance.

## Selection wording
The full development selector scores a fixed eight-candidate grid on strict leave-one-host-out cross-fitted component predictions. A separate 30-row host-omission sensitivity analysis removes one host's rows and re-scores those fixed predictions. The selected branch remained `Conditional_only` under every omission; this supports robustness to omission of a single host's rows but is not independent replication.

## Integrity
`SHA256SUMS.txt` is regenerated from this exact v8 content after cleanup. No `__pycache__` or `.pyc` files are included.

## Public repository record
The Zenodo DOI is created by Zenodo at deposit time and is therefore not embedded in this pre-deposit archive. Replace `[ZENODO DOI TO INSERT BEFORE RESUBMISSION]` in the manuscript/response/cover letter with the live DOI before resubmission.

## License and citation
Code is released under the MIT License. Citation metadata are in `CITATION.cff`; Zenodo metadata are in `.zenodo.json`.
