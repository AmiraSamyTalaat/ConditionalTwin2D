# Final Q1 repair completion report

## Technical status
All strengthening analyses that can be executed with the frozen available data are complete.

## Corrected calibration
The previous invalid Beta fallback was removed. All nested folds now use constrained monotonic calibrators with no silent identity substitution. Temperature scaling was selected by a development-only mean-rank rule across record-weighted and macro-host Brier, NLL, and ECE. Beta retained the lowest record-weighted Brier (0.1176), while temperature had the best macro-host Brier (0.0968). Paired confidence intervals and Holm-adjusted Wilcoxon tests do not establish universal superiority of one calibrator.

## Target robustness
Across 0.05-0.50 micro-B thresholds, internal AUROC remained 0.9137-0.9190. Threshold changes did not restore reliable external within-host ranking.

## Continuous target
Development global and host-centered Spearman correlations were 0.7829 and 0.7160. External correlations remained weak, confirming that binary thresholding alone does not explain transport failure.

## Host shortcut
At 0.10 micro-B, 2DMD pooled AUROC was 0.7754, while host-centered AUROC was 0.5017. The explicit between-host mean-score audit exceeded the primary pooled AUROC and produced chance within-host ranking, demonstrating a material host shortcut.

## Applicability domain
The final AD uses cross-host kNN distance, shrinkage Mahalanobis distance, robust range exceedance, Isolation Forest anomaly score, and ensemble uncertainty. It was fitted and thresholded using development only. It measured support shift, but external AD-error and host-failure predictiveness were weak or inconsistent. This negative result is retained.

## Scientific position
- Strong validation/auditing contribution: yes.
- Strong algorithmic novelty: no; moderate.
- Ready for an honest specialized-Q1 submission attempt after author metadata and repository archiving: yes.
- Strong-Q1 acceptance guarantee: no.
- Further tuning on Imp2D or 2DMD: forbidden.
- Larger independent same-regime lockbox: not available; not fabricated.
