from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
I=ROOT/'base_project'/'inputs'
D=pd.read_csv(I/'development_representation.csv')
F=sorted([c for c in D if c.startswith('x_')]+[c for c in D if c.startswith('c_') and 'delta' in c])
assert len(F)==68
def check(name,df,idcol):
    a=df[F].to_numpy(float)
    print(name, 'records=',len(df),'NaN=',int(np.isnan(a).sum()),'Inf=',int(np.isinf(a).sum()),'duplicate_ids=',int(df[idcol].duplicated().sum()))
check('Development',D,'record_id')
e2=pd.read_csv(I/'2dmd_evaluated_original.csv'); f2=pd.read_csv(I/'2dmd_features.csv'); ids=set(e2.loc[e2.strict_primary.astype(bool),'_id']); check('2DMD strict',f2[f2._id.isin(ids)],'_id')
fi=pd.read_csv(I/'imp2d_features.csv'); check('Imp2D strict',fi[~fi.selected_phase_uid.isin(set(D.host_key))],'external_id')
fq=pd.read_csv(I/'qpod_features.csv'); check('QPOD extension',fq,'record_id')
