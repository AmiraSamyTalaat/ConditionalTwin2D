# Feature and leakage rules

## Allowed raw structure fields

- atomic numbers;
- positions;
- cell;
- periodic boundary flags.

## Forbidden model inputs

- total magnetic moment;
- atomic magnetic moments;
- total energy;
- forces;
- any target-derived field;
- host identity;
- record ID;
- URL or file path.

## Graph construction

The graph is a deterministic geometric neighbor graph based on a scaled sum of
covalent radii, clipped to a physically bounded range. Two-dimensional periodic
images in x and y are considered.

## Twin-difference representation

All direct differences are computed between intensive or normalized
descriptors. Atom count and area ratios are retained separately. No atom-level
correspondence is claimed in Stage 2B.

## Interpretation

Stage 2B tests whether structure-derived signal exists beyond metadata. It is
not an equivariant neural network and is not the final DefectTwin-2D model.
