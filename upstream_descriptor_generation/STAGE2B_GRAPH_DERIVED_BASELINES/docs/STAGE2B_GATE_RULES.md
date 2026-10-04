# Stage 2B gate rules

## GREEN

A structural feature set must:

- achieve binary AUROC at least 0.65;
- achieve ordinal QWK at least 0.30;
- improve binary AUROC over metadata by at least 0.02;
- improve ordinal QWK over metadata by at least 0.05.

GREEN permits:

`STAGE2C_GNN_BASELINE_PILOT`

## YELLOW

The graph pipeline is valid and a structural feature set is no worse than
metadata by more than 0.02 on either principal metric, but the GREEN
improvement thresholds are not both met.

YELLOW permits only a tightly controlled small GNN pilot.

## RED

All structural representations materially underperform metadata, or graph
extraction fails for a meaningful fraction of records.

No Stage 2B decision permits final CDT-EGNN training.
