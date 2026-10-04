# START HERE - Reviewer 2 reproducibility package

Manuscript title: Host Conditional Machine Learning Supports Candidate Ranking of Magnetic Point Defects Within Two Dimensional Materials Using Computed Magnetic Moments

This is the canonical v8 code-and-reproducibility package prepared to answer Reviewer 2's request for public source code and data. The revised manuscript preserves the submitted paper's original 12 figures and 9 tables and adds reviewer-requested Figures 13-15 and Tables 10-12; the detailed machine-readable evidence for all of these items is indexed here.

## One command
```bash
python verify_repository.py
```
This performs package integrity checks, verifies the corrected DefectTwin base package, verifies the frozen ConditionalTwin package, reruns the fast/medium reviewer-facing audits in an isolated temporary copy, and validates the archived outputs of the computationally expensive mechanism, pool-size, and scale-ablation jobs. It does not modify the deposited archive or invalidate its hashes.

Use `python verify_repository.py --quick` for integrity and frozen-package checks only. Use `python verify_repository.py --full-recompute` to rerun every reviewer-facing audit from scratch in an isolated temporary copy; this is computationally expensive.

## Main code
- `base_project/scripts/run_transport_rank.py` - ConditionalTwin-2D transport/ranking workflow.
- `base_defecttwin_final_q1/scripts/` - corrected DefectTwin base analysis.
- `revision_scripts/` - metric audit, 2DMD host audit, PCA, scale-ratio ablation, candidate-pool completeness, graph provenance, pool-size sensitivity, and mechanism/dimensionality analyses.

## Interpretation warning
The 30 host-row-exclusion sensitivity tables re-score fixed leave-one-host-out cross-fitted component predictions. They are not independent replications and are not nested outer-fold model-selection estimates.

## Manuscript provenance
- `MAIN_ITEM_PROVENANCE_MAP.csv` maps every main-manuscript figure and table with explicit model/version, cohort/split, endpoint/threshold, seed/ensemble, calibration, and artifact fields.
- `COHORT_COMPUTATIONAL_PROVENANCE.csv` records source computational settings, charge-state information where available, relaxation, supercell/separation context, filtering/overlap, and explicitly marked unavailable fields.
- `REVIEWER_SUPPORT_INDEX.md` maps reviewer requests to audit outputs.
- `MANUSCRIPT_RESULT_MAP.md` summarizes the preserved-original numbering.

## Public deposit
Upload this exact archive to Zenodo (or another stable public repository), publish the record, then insert the live DOI/URL into the manuscript, response to Reviewer 2, and cover letter before resubmission.
