# ConditionalTwin-2D method card

## Intended use
Ranking multiple defect candidates belonging to the same 2D host, with emphasis on conditional within-host ordering.

## Inputs
- 68 frozen pristine-defect/compositional descriptors per candidate.
- A host identifier used only to form the candidate pool; the literal identifier is not a model feature.

## Representation
For each host candidate pool:
```text
[raw 68 features, raw features - label-blind host-pool mean]
```

## Training weights
- Equal total weight per host.
- Within a mixed host, half of the host weight is assigned to each class.

## Development selection
- Component predictions are strict leave-one-host-out cross-fitted predictions.
- The full development selector scores a fixed eight-candidate grid on those predictions.
- The 30 host-row-exclusion tables remove one host's rows and re-score the same fixed cross-fitted component predictions; components are not refitted after the designated omission.
- Primary criterion: within-host pair AUROC; secondary: host-centered AUROC; then pooled/macro-host AUROC within fixed tolerances.

## Final selected branch
`Conditional_only` was selected in the full development table and retained in all 30 overlapping host-row-exclusion sensitivity tables. This is evidence that selection is not driven by a single omitted host; it is not 30 independent replications and not a nested outer-fold estimate.

## Limitations
- Candidate-pool conditional; not single-record inductive.
- Tiny mixed hosts make worst-host AUROC unstable.
- No untouched external confirmation of ConditionalTwin-2D; external comparisons are retrospective.
