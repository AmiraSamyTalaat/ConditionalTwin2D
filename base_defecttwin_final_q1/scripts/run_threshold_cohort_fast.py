from pathlib import Path
import sys, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1];PREV=ROOT/'outputs/checkpoints';OUT=ROOT/'final_outputs';TABLE=OUT/'tables';CHECK=OUT/'checkpoints';TABLE.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
cohort=sys.argv[1]
def stable_int(x):return int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)
def ece(y,p,w,bins=15):
    y=np.asarray(y,int);p=np.asarray(p,float);w=np.asarray(w,float);w=w/w.sum();edges=np.linspace(0,1,bins+1);out=0.0
    for i in range(bins):
        m=(p>=edges[i])&(p<edges[i+1] if i<bins-1 else p<=edges[i+1])
        if m.any():
            ww=w[m];mass=ww.sum();ww=ww/mass;out+=mass*abs(np.sum(ww*y[m])-np.sum(ww*p[m]))
    return float(out)
def host_stats(g):
    hosts=sorted(g.host.unique());hmap={h:i for i,h in enumerate(hosts)};hidx=g.host.map(hmap).to_numpy(int);y=g.y_true.to_numpy(int);p=g.probability.to_numpy(float);pred=p>=.5;centered=p-g.groupby('host').probability.transform('mean').to_numpy(float)
    h_auc=np.full(len(hosts),np.nan);h_pairs=np.zeros(len(hosts));h_conc=np.zeros(len(hosts))
    for h,i in hmap.items():
        m=hidx==i;yy=y[m];pp=p[m]
        if len(np.unique(yy))==2:h_auc[i]=roc_auc_score(yy,pp)
        pos=pp[yy==1];neg=pp[yy==0]
        if len(pos) and len(neg):
            d=pos[:,None]-neg[None,:];h_conc[i]=(d>0).sum()+.5*(d==0).sum();h_pairs[i]=d.size
    return hosts,hidx,y,p,pred,centered,h_auc,h_pairs,h_conc
def metrics_from_mult(stat,mult):
    hosts,hidx,y,p,pred,centered,h_auc,h_pairs,h_conc=stat;w=mult[hidx].astype(float);mask=w>0;y2=y[mask];p2=p[mask];pred2=pred[mask];c2=centered[mask];w2=w[mask]
    if np.unique(y2).size<2:return None
    auc=roc_auc_score(y2,p2,sample_weight=w2);ap=average_precision_score(y2,p2,sample_weight=w2);ba=balanced_accuracy_score(y2,pred2,sample_weight=w2);br=np.average((y2-p2)**2,weights=w2);ec=ece(y2,p2,w2)
    hc=roc_auc_score(y2,c2,sample_weight=w2)
    ev=np.isfinite(h_auc)&(mult>0);macro=np.average(h_auc[ev],weights=mult[ev]) if ev.any() else np.nan
    pairs=np.sum(h_pairs*mult);within=np.sum(h_conc*mult)/pairs if pairs>0 else np.nan
    return {'AUROC':auc,'AUPRC':ap,'balanced_accuracy':ba,'Brier':br,'ECE15':ec,'macro_host_AUROC':macro,'host_centered_AUROC':hc,'within_host_pair_AUROC':within}
def point(g):
    stat=host_stats(g);mult=np.ones(len(stat[0]),int);m=metrics_from_mult(stat,mult);y=stat[2];m.update({'records':len(g),'hosts':len(stat[0]),'positives':int(y.sum()),'prevalence':float(y.mean())});return m,stat
if cohort=='Development_LOHO':
    d=pd.read_csv(PREV/'threshold_internal_predictions.csv');ens=d.groupby(['endpoint_muB','record_id','host','y_true'],as_index=False).probability.mean();ens['cohort']=cohort
else:
    d=pd.read_csv(PREV/'threshold_external_predictions.csv');d=d[d.cohort.eq(cohort)];ens=d.groupby(['endpoint_muB','record_id','host','y_true'],as_index=False).probability.mean();ens['cohort']=cohort
ens.to_csv(CHECK/f'FINAL_THRESHOLD_ENSEMBLE_{cohort}.csv',index=False)
rows=[];ci=[]
for endpoint,g in ens.groupby('endpoint_muB'):
    m,stat=point(g);rows.append({'cohort':cohort,'endpoint_muB':endpoint,**m});H=len(stat[0]);rng=np.random.default_rng(2026071702+stable_int(f'{cohort}:{endpoint}'));boot=[]
    for b in range(1000):
        sampled=rng.integers(0,H,size=H);mult=np.bincount(sampled,minlength=H);x=metrics_from_mult(stat,mult)
        if x is not None:boot.append(x)
    bt=pd.DataFrame(boot)
    for metric in ['AUROC','AUPRC','balanced_accuracy','Brier','ECE15','macro_host_AUROC','host_centered_AUROC','within_host_pair_AUROC']:
        v=bt[metric].dropna();ci.append({'cohort':cohort,'endpoint_muB':endpoint,'metric':metric,'estimate':m[metric],'ci_2_5':v.quantile(.025) if len(v) else np.nan,'ci_97_5':v.quantile(.975) if len(v) else np.nan,'valid_bootstraps':len(v)})
pd.DataFrame(rows).to_csv(TABLE/f'TABLE_FINAL_THRESHOLD_SENSITIVITY_{cohort}.csv',index=False);pd.DataFrame(ci).to_csv(TABLE/f'TABLE_FINAL_THRESHOLD_CI_{cohort}.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
