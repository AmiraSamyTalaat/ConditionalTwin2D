# Main-manuscript item provenance map

Explicit fields requested during peer review. Each row states model/version, cohort/split, endpoint/threshold, seed/ensemble status, calibration status, and the primary archived prediction/result artifact.

## Figure 1
- **Model/version:** Study-design schematic; no fitted model
- **Cohort/split:** Development plus external cohort roles
- **Endpoint/threshold:** Primary endpoint >=0.10 μB; robustness thresholds separate
- **Seed/ensemble:** N/A
- **Calibration:** N/A
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/Figure1_Final_Workflow.png
- **Notes:** Original workflow schematic; frozen study design/evidence chain.

## Table 1
- **Model/version:** Cohort inventory; no fitted model
- **Cohort/split:** Development 529/30; Imp2D strict 11866/40; 2DMD strict 2500/5; QPOD extension 693/7
- **Endpoint/threshold:** Primary endpoint >=0.10 μB where labels are summarized
- **Seed/ensemble:** N/A
- **Calibration:** N/A
- **Prediction/result artifact:** base_project/inputs/development_targets_manifest.csv; base_defecttwin_final_q1/inputs/*evaluated_original.csv; base_project/inputs/qpod_evaluated.csv
- **Notes:** Original cohort inventory and study roles.

## Table 2
- **Model/version:** Mixed model/control summary; primary DefectTwin-2D ExtraTrees is 68-variable
- **Cohort/split:** Development host-disjoint evaluation
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** ExtraTrees uses fixed seeds 20260715-20260719; model-specific control summaries as archived
- **Calibration:** No probability recalibration in discrimination summary
- **Prediction/result artifact:** base_defecttwin_final_q1/analysis/FINAL_SCIENTIFIC_SUMMARY.json; base_defecttwin_final_q1/tables/DefectTwin2D_Final_Q1_Repair_Tables.xlsx
- **Notes:** Original development model/control summary; GCN rows retained as exploratory controls, not a modern-network superiority claim.

## Table 3
- **Model/version:** DefectTwin-2D 68-variable ExtraTrees calibration audit
- **Cohort/split:** Development outer held-out-host with inner host-grouped calibration
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Frozen base cross-fitted development predictions; five fixed base seeds where applicable
- **Calibration:** Identity, Temperature, Beta, Platt; Temperature selected by prespecified mean-rank rule
- **Prediction/result artifact:** base_defecttwin_final_q1/tables/TABLE_FINAL_CALIBRATION_COMPARISON.csv; checkpoints/FINAL_NESTED_CALIBRATION_PREDICTIONS.csv
- **Notes:** Corrected nested host-disjoint calibration summary for the original 68-variable base model.

## Figure 2
- **Model/version:** DefectTwin-2D calibration comparison
- **Cohort/split:** Development only
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Same frozen base calibration inputs as Table 3
- **Calibration:** Identity, Temperature, Beta, Platt; selected method shown in Table 3
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF0_Calibration_Comparison.png
- **Notes:** Original corrected calibration comparison.

## Figure 3
- **Model/version:** DefectTwin-2D grouped held-out-host permutation audit
- **Cohort/split:** Development held-out-host evaluation
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Frozen base held-out-host predictions; grouped permutation audit
- **Calibration:** No calibrator is selected within the permutation audit; plotted score is held-out-host Brier increase
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF7_Grouped_Permutation_Importance.png; base_defecttwin_final_q1/tables/TABLE_FINAL_GROUPED_PERMUTATION_IMPORTANCE.csv
- **Notes:** Original grouped held-out-host permutation analysis; revised wording uses increase in held-out-host Brier score.

## Figure 4
- **Model/version:** DefectTwin-2D reliability audit
- **Cohort/split:** Development calibration analysis
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Frozen base cross-fitted development predictions
- **Calibration:** Record-weighted reliability for valid calibration candidates
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF1_Record_Weighted_Reliability.png; base_defecttwin_final_q1/tables/TABLE_FINAL_RELIABILITY_DIAGRAM_DATA.csv
- **Notes:** Original record-weighted reliability diagram.

## Figure 5
- **Model/version:** DefectTwin-2D threshold-robustness refits
- **Cohort/split:** Development, 2DMD strict, Imp2D strict
- **Endpoint/threshold:** Thresholds 0.05, 0.10, 0.20, 0.50 μB
- **Seed/ensemble:** Same frozen estimator family and fixed seed set; development-only fitting procedure
- **Calibration:** Robustness analysis; not used for post-lockbox model selection
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF3_Threshold_Sensitivity_CI.png; base_defecttwin_final_q1/tables/TABLE_FINAL_THRESHOLD_SENSITIVITY*.csv
- **Notes:** Original endpoint-threshold sensitivity with confidence intervals.

## Figure 6
- **Model/version:** Continuous-target DefectTwin-2D models
- **Cohort/split:** Development, 2DMD strict, Imp2D strict
- **Endpoint/threshold:** Continuous target log(1+|M|)
- **Seed/ensemble:** Frozen seed-specific continuous predictions aggregated as archived
- **Calibration:** Probability calibration not applicable to continuous target
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF4_Continuous_Global_HostCentered.png; base_defecttwin_final_q1/tables/TABLE_FINAL_CONTINUOUS_TARGET.csv
- **Notes:** Original continuous-target transport correlations.

## Table 4
- **Model/version:** Original prediction-sealed DefectTwin-2D 68-variable base model
- **Cohort/split:** Imp2D strict and 2DMD strict
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Five fixed base seeds; external predictions sealed before target access
- **Calibration:** No external retuning; ranking metrics use sealed base prediction stream
- **Prediction/result artifact:** base_defecttwin_final_q1/inputs/imp2d_evaluated_original.csv; base_defecttwin_final_q1/inputs/2dmd_evaluated_original.csv
- **Notes:** Original prediction-sealed DefectTwin-2D transport metrics.

## Table 5
- **Model/version:** DefectTwin-2D primary fixed predictions plus host-constant shortcut audits
- **Cohort/split:** 2DMD strict
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Fixed sealed predictions; no refit for host-constant audits
- **Calibration:** No new calibration fit; tie-safe centering audit applied to host-constant scores
- **Prediction/result artifact:** base_defecttwin_final_q1/tables/TABLE_FINAL_EXPLICIT_BASELINES.csv; revision_results/TIE_SAFE_HOST_CENTERED_AUDIT.csv
- **Notes:** Original 2DMD host-shortcut/prevalence audit with reviewer-requested tie-safe correction of host-constant host-centered AUROC.

## Figure 7
- **Model/version:** DefectTwin-2D primary/between-host score audit
- **Cohort/split:** Development / 2DMD / Imp2D as labeled
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Fixed archived score streams
- **Calibration:** No refit; descriptive score comparison
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF5_Explicit_Between_Host_Baseline.png; base_defecttwin_final_q1/tables/TABLE_FINAL_EXPLICIT_BASELINES.csv
- **Notes:** Original primary versus between-host-only scores.

## Table 6
- **Model/version:** DefectTwin-2D 68-variable applicability-domain audit
- **Cohort/split:** Development-frozen AD applied to development, 2DMD, Imp2D
- **Endpoint/threshold:** Primary endpoint only for error/performance summaries; AD itself is label-free
- **Seed/ensemble:** Five-seed base ExtraTrees uncertainty is one AD component
- **Calibration:** AD is not a probability calibrator; fitted on development descriptors only
- **Prediction/result artifact:** base_defecttwin_final_q1/tables/TABLE_FINAL_AD_V2_COHORT_SUMMARY.csv
- **Notes:** Original five-component applicability-domain cohort summary for the 68-variable base space.

## Figure 8
- **Model/version:** DefectTwin-2D 68-variable applicability-domain distributions
- **Cohort/split:** Development-frozen AD
- **Endpoint/threshold:** N/A for AD score
- **Seed/ensemble:** Same frozen AD as Table 6
- **Calibration:** N/A
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF6_AD_V2_Distribution.png
- **Notes:** Original applicability-domain score distributions.

## Figure 9
- **Model/version:** DefectTwin-2D applicability-domain retention audit
- **Cohort/split:** Development, 2DMD, Imp2D retained subsets
- **Endpoint/threshold:** Primary endpoint >=0.10 μB for AUROC after retention
- **Seed/ensemble:** Fixed archived base predictions; no refit
- **Calibration:** No recalibration during retention; composition changes are interpreted descriptively
- **Prediction/result artifact:** base_defecttwin_final_q1/figures/FigureF7_AD_V2_Retention.png; base_defecttwin_final_q1/tables/TABLE_FINAL_AD_V2_RETENTION.csv
- **Notes:** Original applicability-domain retention analysis.

## Table 7
- **Model/version:** ConditionalTwin-2D development ablations
- **Cohort/split:** Development strict LOHO cross-fitted component predictions
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** ExtraTrees components use fixed seeds 20260715-20260719; finite selector on cross-fitted predictions
- **Calibration:** No probability calibration used for within-host ranking selection
- **Prediction/result artifact:** base_project/outputs/tables/DEVELOPMENT_CROSSFIT_PREDICTIONS.csv; base_project/outputs/tables/DEVELOPMENT_FULL_SELECTION_TABLE.csv
- **Notes:** Original ConditionalTwin development ablations.

## Figure 10
- **Model/version:** ConditionalTwin-2D development comparison
- **Cohort/split:** Development only
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Same fixed development component predictions as Table 7
- **Calibration:** No probability calibration used for ranking comparison
- **Prediction/result artifact:** base_project/outputs/figures/FIG1_DEVELOPMENT_CONDITIONALTWIN_COMPARISON.png
- **Notes:** Original development comparison of ET and ConditionalTwin ablations.

## Table 8
- **Model/version:** Retrospective ConditionalTwin-2D transport comparison with workflow-specific ET comparator
- **Cohort/split:** Development LOHO plus retrospective 2DMD, Imp2D, QPOD
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Development-fitted fixed-seed models; no external tuning
- **Calibration:** No external recalibration; values must not be substituted with sealed Table 4 or threshold-refit Section 3.3 values
- **Prediction/result artifact:** base_project/outputs/tables/DEVELOPMENT_CROSSFIT_PREDICTIONS.csv; base_project/outputs/tables/2DMD_RETROSPECTIVE_PREDICTIONS.csv; base_project/outputs/tables/IMP2D_RETROSPECTIVE_PREDICTIONS.csv; base_project/outputs/tables/QPOD_RETROSPECTIVE_PREDICTIONS.csv
- **Notes:** Original retrospective transport comparison; revised note clarifies workflow-specific ET values.

## Figure 11
- **Model/version:** ConditionalTwin-2D retrospective within-host transport matrix
- **Cohort/split:** Development and retrospective external cohorts
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Same fixed prediction files as Table 8
- **Calibration:** No external recalibration
- **Prediction/result artifact:** base_project/outputs/figures/FIG2_WITHIN_HOST_TRANSPORT_MATRIX.png
- **Notes:** Original within-host transport matrix.

## Figure 12
- **Model/version:** Descriptive pooled-minus-within-host shortcut index
- **Cohort/split:** Development and retrospective external cohorts
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Derived from fixed Table 8 prediction metrics
- **Calibration:** N/A; descriptive derived index
- **Prediction/result artifact:** base_project/outputs/figures/FIG3_SHORTCUT_AMPLIFICATION.png
- **Notes:** Original shortcut amplification index.

## Table 9
- **Model/version:** Evidence-scope synthesis; no new fit
- **Cohort/split:** Synthesis of development and external analyses
- **Endpoint/threshold:** As stated per evidence row
- **Seed/ensemble:** N/A
- **Calibration:** N/A
- **Prediction/result artifact:** base_project/outputs/CLAIM_BOUNDARIES.md; revision_results/REVIEWER_CLOSURE_AUDIT.md
- **Notes:** Original evidence-scope table, revised conservatively to reflect retrospective/heterogeneous external evidence and exploratory GCN controls.

## Table 10
- **Model/version:** ConditionalTwin-2D mechanism/effective-dimensionality audit
- **Cohort/split:** Development frozen representation and predictions
- **Endpoint/threshold:** Primary endpoint >=0.10 μB for permutation/AUROC components
- **Seed/ensemble:** Five final ExtraTrees seeds 20260715-20260719; 108,403 total internal splits
- **Calibration:** No probability calibration; ranking/mechanism audit
- **Prediction/result artifact:** revision_results/MECHANISM_DIMENSIONALITY_SPLITS_SUMMARY.json; revision_results/MECHANISM_FEATURE_PERMUTATION.csv; revision_results/SCALE_RATIO_ABLATION.csv
- **Notes:** Reviewer-requested representation geometry and mechanism audit.

## Table 11
- **Model/version:** Leave-one-out ConditionalTwin candidate-pool sensitivity
- **Cohort/split:** Development mixed-class hosts; n=2,3,5,10,full
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Frozen estimator family/fixed seed set; leave-one-out host mean; host-block bootstrap
- **Calibration:** No probability calibration; within-host ranking audit
- **Prediction/result artifact:** revision_results/LOO_POOL_SENSITIVITY_SUMMARY.csv; revision_results/LOO_POOL_SENSITIVITY_BOOTSTRAP_CI.csv
- **Notes:** Reviewer-requested leave-one-out candidate-pool sensitivity with host-block CIs.

## Figure 13
- **Model/version:** Leave-one-out ConditionalTwin candidate-pool sensitivity plot
- **Cohort/split:** Development mixed-class hosts
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Same predictions/bootstrap as Table 11
- **Calibration:** No probability calibration
- **Prediction/result artifact:** revision_results/LOO_POOL_SENSITIVITY_SUMMARY.csv; revision_results/LOO_POOL_SENSITIVITY_BOOTSTRAP_CI.csv
- **Notes:** Reviewer-requested leave-one-out candidate-pool sensitivity plot.

## Figure 14
- **Model/version:** PCA of original 68 development descriptors colored by log(1+|M|)
- **Cohort/split:** Development only
- **Endpoint/threshold:** Continuous color = log(1+absolute magnetic moment)
- **Seed/ensemble:** No model seed; deterministic StandardScaler/PCA(random_state=0)
- **Calibration:** N/A
- **Prediction/result artifact:** revision_results/PCA_BY_MAGNETIC_MOMENT.png; revision_results/PCA_SUMMARY.json; revision_results/PCA_DEVELOPMENT_COORDINATES.csv
- **Notes:** Reviewer-requested PCA of 68 development descriptors colored by log(1+|moment|).

## Figure 15
- **Model/version:** PCA of original 68 development descriptors colored by host
- **Cohort/split:** Development only
- **Endpoint/threshold:** No binary endpoint used for coloring
- **Seed/ensemble:** No model seed; deterministic StandardScaler/PCA(random_state=0)
- **Calibration:** N/A
- **Prediction/result artifact:** revision_results/PCA_BY_HOST.png; revision_results/PCA_SUMMARY.json; revision_results/PCA_DEVELOPMENT_COORDINATES.csv
- **Notes:** Reviewer-requested PCA of 68 development descriptors colored by host.

## Table 12
- **Model/version:** Fixed-prediction 2DMD per-host ET versus ConditionalTwin influence audit
- **Cohort/split:** 2DMD retrospective; BP/GaSe/InSe class-complete; MoS2/WSe2 single-class
- **Endpoint/threshold:** Primary endpoint >=0.10 μB
- **Seed/ensemble:** Uses fixed workflow-specific ET/Conditional predictions from retrospective comparison; bootstrap does not refit
- **Calibration:** No recalibration or model selection repeated in host bootstrap
- **Prediction/result artifact:** revision_results/2DMD_PER_HOST_AUDIT.csv; revision_results/2DMD_HOST_AUDIT_SUMMARY.csv
- **Notes:** Reviewer-requested fixed-prediction 2DMD per-host influence audit.
