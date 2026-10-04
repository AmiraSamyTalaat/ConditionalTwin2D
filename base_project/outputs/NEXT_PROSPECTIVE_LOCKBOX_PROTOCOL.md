# Prospective confirmation protocol

1. Freeze the exact feature list, ConditionalTwin weights, preprocessing, seeds, primary threshold, and calibration.
2. Register an untouched source before reading magnetic targets.
3. Require record-disjoint and exact-phase-disjoint data; composition-unseen is preferred and must be named accurately.
4. Seal classification and continuous predictions with SHA-256 before target access.
5. Report all hosts, including below-chance hosts.
6. Primary metrics: within-host pair AUROC, host-centered AUROC, pooled AUROC, Brier skill, and top-10% within-host enrichment.
7. No external recalibration, threshold change, score inversion, feature selection, or host deletion.
8. A DFT mini-lockbox of 5–8 untouched hosts and 20–30 defects per host is acceptable if no suitable public database exists.
