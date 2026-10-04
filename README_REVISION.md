## Canonical repository package
This v8 directory is the single canonical public-deposit package. Do not deposit older revision archives alongside it.

# ConditionalTwin-2D Scientific Reports revision reproducibility package

**Revised manuscript title**

Host Conditional Machine Learning Supports Candidate Ranking of Magnetic Point Defects Within Two Dimensional Materials Using Computed Magnetic Moments

## Scope

This package contains the frozen descriptor matrices, target/cohort manifests, prediction files, model-selection outputs, analysis code, result hashes, the additional scripts used for the Scientific Reports revision, and the recovered original Stage 2B graph-descriptor source/configuration with its Stage 2A structure input. It reproduces the manuscript analyses from the archived 68-variable descriptor representation onward and documents the raw-structure graph construction used for the 52 graph-derived pristine-defect variables.

The recovered Stage 2B configuration defines a pair-specific neighbor threshold as `clip(1.25 * (r_cov_i + r_cov_j), 1.1 A, 3.5 A)` and evaluates the minimum distance over in-plane translations `tx,ty in {-1,0,1}`, with no z translation. The provenance audit in this package recomputes the 52 graph-derived variables for one record from each of all 30 development hosts and matches the final development matrix within floating-point precision.

## Directory layout

- `base_project/` — frozen transport/ranking project with inputs, original analysis code, configuration, saved outputs, and verification files.
- `revision_scripts/` — additional analyses requested during peer review.
- `revision_results/` — numerical tables, plots, logs, and JSON summaries produced for the revision.
- `upstream_descriptor_generation/` — recovered original Stage 2B graph-descriptor source/configuration and Stage 2A structure input.
- `SHA256SUMS.txt` — hashes of package contents (excluding the hash file itself).

## Environment

Python 3.11+ is recommended. Install the base requirements with:

```bash
python -m pip install -r base_project/requirements.txt
```

The package uses NumPy, pandas, SciPy, scikit-learn, matplotlib, and PyYAML.

## Verify the frozen base project

From the package root:

```bash
python base_project/scripts/verify_stage.py
```

Expected terminal status includes `VERIFICATION PASSED`.

The primary base workflow is:

```bash
python base_project/scripts/run_transport_rank.py
```

This can overwrite/regenerate files under `base_project/outputs/`; keep a copy of the archived package if preservation of the frozen outputs is required.

## Revision analyses

Each revision script resolves paths relative to this package root and writes to `revision_results/`.

### 1. Tie-safe host-centered metric audit

```bash
python revision_scripts/run_metric_tie_audit.py
```

Produces `TIE_SAFE_HOST_CENTERED_AUDIT.csv`. The audit sets same-host centered residuals with absolute magnitude below `1e-12` to exact zero before AUROC computation. Host-constant baselines then correctly return host-centered AUROC 0.5000. The primary nonconstant ET and ConditionalTwin scores are included in the same audit to show that their results are unchanged.

### 2. 2DMD per-host influence analysis

```bash
python revision_scripts/run_2dmd_host_audit.py
```

Produces `2DMD_PER_HOST_AUDIT.csv` and `2DMD_HOST_AUDIT_SUMMARY.csv`. The bootstrap is a fixed-prediction host-block bootstrap over the three informative 2DMD hosts; it does not repeat model fitting or selection.

### 3. Leave-one-out candidate-pool sensitivity

```bash
python revision_scripts/run_loo_pool_sensitivity.py
```

Produces the pool-size summary, by-host table, full-pool predictions, and 5,000-replicate host-block confidence intervals for `n = 2, 3, 5, 10, full`. The conditioning mean for each record excludes that record.

### 4. Conditional-representation mechanism and effective dimensionality

```bash
python revision_scripts/run_mechanism_audit.py
```

Produces the standardized-rank/participation-ratio/split-share summary and raw-versus-centered held-out-host permutation audit. This is the most computationally expensive revision script.

### 5. PCA projection audit

```bash
python revision_scripts/run_pca_audit.py
```

Produces PCA coordinates and projections colored by computed magnetic moment and host, plus explained-variance and silhouette summaries.

### 6. Atom-count/area-ratio ablation

```bash
python revision_scripts/run_scale_ratio_ablation.py
```

Re-evaluates the conditional representation after removing `x_atom_count_ratio` and `x_area_ratio`.


### 7. Graph-construction provenance audit

```bash
python revision_scripts/verify_graph_provenance.py
```

Uses the recovered original Stage 2B source and Stage 2A structures to recompute the 52 graph-derived pristine-defect variables for one record from each of the 30 development hosts and compare them with `base_project/inputs/development_representation.csv`. The archived audit contains 1,560 numerical comparisons, zero differences above `1e-10`, and a maximum absolute difference of approximately `3.55e-15`.

### 8. Candidate-pool completeness and exact selection protocol

```bash
python revision_scripts/run_candidate_pool_completeness_audit.py
```

The archived `CANDIDATE_POOL_COMPLETENESS_AUDIT.csv` verifies zero NaN, zero infinite values, and zero duplicate primary IDs across the 68 locked descriptors in all four analyzed cohorts. `DEVELOPMENT_VARIANT_DUPLICATE_AUDIT.json` documents source-distinct development variants and confirms no duplicate record IDs, detail URLs, or defect-structure hashes. `FEATURE_DICTIONARY_68.csv` is the full locked descriptor dictionary. `MODEL_SELECTION_PROTOCOL.md` records the exact eight-candidate grid, weighting formulas, tolerances, tie-breaking, and final fitting procedure.

The ConditionalTwin component predictions are strict leave-one-host-out predictions. The 30 host-row-exclusion tables are selection-sensitivity analyses that re-score fixed cross-fitted predictions; they are not nested outer-fold selection estimates. The separate calibration analysis retains its true nested three-fold host-group procedure.

## Key archived revision results

The `revision_results/` directory includes the results used in the revised manuscript and response letter, including:

- 136 nominal variables, 131 nonconstant variables, standardized rank 86, participation ratio 23.6634, 40 PCs for 95% variance, and 55 PCs for 99% variance.
- 108,403 final ExtraTrees splits across five seeds, of which 53,904 (49.73%) use centered variables.
- Leave-one-out candidate-pool sensitivity showing positive ConditionalTwin-minus-ET within-host AUROC differences at `n=2,3,5,10,full`; the small-pool support is explicitly limited by the number of eligible mixed-class hosts.
- Tie-safe correction of host-constant host-centered AUROC values to 0.5000 without changing primary ET or ConditionalTwin nonconstant-score results.
- 2DMD host-level analysis showing heterogeneous retrospective behavior: the aggregate increase is driven mainly by BP, while GaSe and InSe decrease slightly.
- PCA projection and a scale-related-feature ablation.
- Graph provenance audit: all 30 development hosts sampled, 52 graph-derived variables per host, 1,560 comparisons, zero differences above 1e-10, maximum absolute difference about 3.55e-15.
- Candidate-pool completeness audit: zero NaN, zero infinities, and zero duplicate primary IDs across all 68 locked descriptors in development, 2DMD strict, Imp2D strict, and the QPOD extension.
- Exact finite selection protocol and full 68-descriptor dictionary archived under `revision_results/`.

## Evidence scope

The prediction-sealed status applies to the original base DefectTwin-2D external predictions. ConditionalTwin-2D was developed after those external outcomes were opened, so its 2DMD, Imp2D, and QPOD comparisons are retrospective cross-cohort diagnostics rather than new prospective lockboxes.

## Graph provenance status

Resolved. The original Stage 2B graph-generation source/configuration and Stage 2A structure input are included under `upstream_descriptor_generation/`, and the recovered graph settings have been verified against the final development representation. The remaining public-deposit step is administrative: upload the package to the selected repository and insert the resulting URL/DOI in the manuscript and submission system.


## Recovered corrected base analysis
`base_defecttwin_final_q1/` contains the preserved corrected DefectTwin base-analysis source, frozen inputs, exact calibration/threshold/continuous-target checkpoints, AD outputs, tables, and figures recovered from the evidence chain. `MAIN_ITEM_PROVENANCE_MAP.csv` maps every main manuscript table and figure with explicit model/version, cohort/split, endpoint/threshold, seed/ensemble, calibration status, and primary artifact. `COHORT_COMPUTATIONAL_PROVENANCE.csv` records the source computational and filtering/overlap context without inferring unavailable concentration or charge-state fields. The package is ready for public repository deposition; insert the resulting stable URL/DOI into the manuscript before resubmission.
