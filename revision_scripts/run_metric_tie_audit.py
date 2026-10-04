from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG / 'base_project'
OUT = PKG / 'revision_results'
OUT.mkdir(exist_ok=True)

TOL = 1e-12

def host_centered_auc(frame, score, tie_safe=True):
    x = pd.DataFrame({'host': frame['host'].astype(str), 'y': frame['y'].astype(int), 'score': np.asarray(score, float)})
    centered = x['score'] - x.groupby('host')['score'].transform('mean')
    if tie_safe:
        centered = centered.mask(centered.abs() < TOL, 0.0)
    return float(roc_auc_score(x['y'], centered))

def within_host_auc(frame, score):
    x = pd.DataFrame({'host': frame['host'].astype(str), 'y': frame['y'].astype(int), 'score': np.asarray(score, float)})
    concordant = 0.0
    pairs = 0
    for _, g in x.groupby('host'):
        if g['y'].nunique() < 2:
            continue
        p = g.loc[g.y.eq(1), 'score'].to_numpy(float)
        n = g.loc[g.y.eq(0), 'score'].to_numpy(float)
        d = p[:, None] - n[None, :]
        concordant += float((d > 0).sum() + 0.5 * (d == 0).sum())
        pairs += int(d.size)
    return float(concordant / pairs) if pairs else np.nan

files = {
    'Development': ROOT / 'outputs/tables/DEVELOPMENT_CROSSFIT_PREDICTIONS.csv',
    '2DMD': ROOT / 'outputs/tables/2DMD_RETROSPECTIVE_PREDICTIONS.csv',
    'Imp2D': ROOT / 'outputs/tables/IMP2D_RETROSPECTIVE_PREDICTIONS.csv',
}

dev = pd.read_csv(files['Development'])
dev_prevalence = float(dev['y'].mean())
rows = []
for cohort, path in files.items():
    df = pd.read_csv(path)
    # Host-constant baselines: after same-host centering these must be exact ties.
    baselines = {
        'global_development_prevalence': np.full(len(df), dev_prevalence),
        'oracle_host_prevalence': df.groupby('host')['y'].transform('mean').to_numpy(float),
    }
    # Add the actual nonconstant model scores to prove core values are unchanged by the tie rule.
    for col in ['ET_score', 'ConditionalET_score']:
        if col in df.columns:
            baselines[col] = df[col].to_numpy(float)
    for label, score in baselines.items():
        rows.append({
            'cohort': cohort,
            'score': label,
            'host_centered_auc_naive': host_centered_auc(df, score, tie_safe=False),
            'host_centered_auc_tie_safe': host_centered_auc(df, score, tie_safe=True),
            'within_host_auc': within_host_auc(df, score),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT / 'TIE_SAFE_HOST_CENTERED_AUDIT.csv', index=False)
print(out.to_string(index=False))
