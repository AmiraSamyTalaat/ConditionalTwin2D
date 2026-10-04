from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
dev=pd.read_csv(ROOT/'inputs/development_representation.csv')
target=pd.read_csv(ROOT/'inputs/development_targets_manifest.csv')
d2=pd.read_csv(ROOT/'inputs/2dmd_features.csv')
imp=pd.read_csv(ROOT/'inputs/imp2d_features.csv')
features=sorted([c for c in dev if c.startswith('x_')]+[c for c in dev if c.startswith('c_') and 'delta' in c])
assert len(dev)==529 and dev.host_key.nunique()==30
assert len(target)==529 and target.record_id.nunique()==529
assert len(features)==68
assert len(d2)==3000
assert len(imp)==12813
print('Self-test PASS: 529 development rows, 30 hosts, 68 features, 3000 2DMD rows, 12813 Imp2D rows')
