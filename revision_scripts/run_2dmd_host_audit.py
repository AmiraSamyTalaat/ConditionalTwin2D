from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG / 'base_project'
OUT = PKG / 'revision_results'
OUT.mkdir(exist_ok=True)

pred = pd.read_csv(ROOT / 'outputs/tables/2DMD_RETROSPECTIVE_PREDICTIONS.csv')
rows = []
for host, g in pred.groupby('host', sort=True):
    pos = int(g['y'].sum())
    neg = int(len(g) - pos)
    pairs = pos * neg
    if pairs:
        et = float(roc_auc_score(g['y'], g['ET_score']))
        cond = float(roc_auc_score(g['y'], g['ConditionalET_score']))
        delta = cond - et
    else:
        et = cond = delta = np.nan
    rows.append({'host': host, 'n': len(g), 'positive': pos, 'negative': neg, 'pairs': pairs,
                 'ET_AUROC': et, 'Conditional_AUROC': cond, 'delta': delta})

t = pd.DataFrame(rows)
t['pair_weight'] = t['pairs'] / t['pairs'].sum()
t.to_csv(OUT / '2DMD_PER_HOST_AUDIT.csv', index=False)

inf = t[t['pairs'] > 0].copy()
w = inf['pairs'].to_numpy(float)
et = float(np.sum(inf['ET_AUROC'].to_numpy(float) * w) / w.sum())
cond = float(np.sum(inf['Conditional_AUROC'].to_numpy(float) * w) / w.sum())

without_bp = inf[inf['host'] != 'BP']
wb = without_bp['pairs'].to_numpy(float)
et_no_bp = float(np.sum(without_bp['ET_AUROC'].to_numpy(float) * wb) / wb.sum())
cond_no_bp = float(np.sum(without_bp['Conditional_AUROC'].to_numpy(float) * wb) / wb.sum())

# Fixed-prediction host-block bootstrap over the informative hosts only.
rng = np.random.default_rng(20261002)
boot = []
for _ in range(5000):
    sb = inf.iloc[rng.integers(0, len(inf), len(inf))]
    ww = sb['pairs'].to_numpy(float)
    e = float(np.sum(sb['ET_AUROC'].to_numpy(float) * ww) / ww.sum())
    c = float(np.sum(sb['Conditional_AUROC'].to_numpy(float) * ww) / ww.sum())
    boot.append((e, c, c-e))
boot = np.asarray(boot, float)
q = np.quantile(boot, [0.025, 0.975], axis=0)
summary = pd.DataFrame([{
    'informative_hosts': len(inf),
    'ET_within_host_AUROC': et,
    'Conditional_within_host_AUROC': cond,
    'delta': cond-et,
    'delta_ci_low': q[0,2],
    'delta_ci_high': q[1,2],
    'delta_without_BP': cond_no_bp-et_no_bp,
    'bootstrap_replicates': 5000,
    'bootstrap_refits_model': False,
}])
summary.to_csv(OUT / '2DMD_HOST_AUDIT_SUMMARY.csv', index=False)
print(t.to_string(index=False))
print('\n', summary.to_string(index=False))
