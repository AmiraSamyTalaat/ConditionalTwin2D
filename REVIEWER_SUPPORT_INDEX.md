# Reviewer-support artifact index

The revised Scientific Reports manuscript preserves the submitted paper and adds the requested audits directly to the main text. Detailed machine-readable evidence is deposited here with the public code/data record.

- Reviewer 1 mechanism/dimensionality (Main Table 10): `revision_results/MECHANISM_DIMENSIONALITY_SPLITS_SUMMARY.json`, `MECHANISM_FEATURE_PERMUTATION.csv`, `SCALE_RATIO_ABLATION.csv`
- Reviewer 1 pool-size sensitivity (Main Table 11 and Figure 13): `LOO_POOL_SENSITIVITY_SUMMARY.csv`, `LOO_POOL_SENSITIVITY_BOOTSTRAP_CI.csv`, `LOO_POOL_SENSITIVITY_BY_HOST.csv`
- Reviewer 3 PCA (Main Figures 14-15): `PCA_BY_MAGNETIC_MOMENT.png/pdf`, `PCA_BY_HOST.png/pdf`, `PCA_SUMMARY.json`, `PCA_DEVELOPMENT_COORDINATES.csv`
- Reviewer 3 graph provenance: `GRAPH_PROVENANCE_AUDIT_SUMMARY.json`, `GRAPH_PROVENANCE_AUDIT_BY_HOST.csv`, plus `upstream_descriptor_generation/`
- Reviewer 3 scale-feature ablation (reported in Table 10): `SCALE_RATIO_ABLATION.csv`
- Reviewer 4 tie-safe host centering (corrected Main Table 5): `TIE_SAFE_HOST_CENTERED_AUDIT.csv`
- Reviewer 4 2DMD per-host influence (Main Table 12): `2DMD_PER_HOST_AUDIT.csv`, `2DMD_HOST_AUDIT_SUMMARY.csv`
- Reviewer 4 candidate-pool completeness and descriptor dictionary: `CANDIDATE_POOL_COMPLETENESS_AUDIT.csv`, `FEATURE_DICTIONARY_68.csv`, `DEVELOPMENT_VARIANT_DUPLICATE_AUDIT.json`
- Exact selector/weighting/tie-breaking: `MODEL_SELECTION_PROTOCOL.md`
- Main-item provenance with explicit model/split/threshold/seed/calibration/artifact fields: `MAIN_ITEM_PROVENANCE_MAP.csv`, `MAIN_ITEM_PROVENANCE_MAP.md`
- Cohort computational/physical provenance: `COHORT_COMPUTATIONAL_PROVENANCE.csv`, `COHORT_COMPUTATIONAL_PROVENANCE.md`
