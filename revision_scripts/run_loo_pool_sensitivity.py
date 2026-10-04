from __future__ import annotations
import sys, math, itertools, hashlib
from pathlib import Path
import numpy as np, pandas as pd, yaml
PKG=Path(__file__).resolve().parents[1]
ROOT=PKG/'base_project'
OUT=PKG/'revision_results'
sys.path.insert(0,str(ROOT/'scripts'))
import run_transport_rank as rr
cfg=yaml.safe_load((ROOT/'config/project_config.yaml').read_text())

def loo_center(frame, features):
    X=frame[features].astype(float).copy()
    sums=frame.groupby('host')[features].transform('sum').astype(float)
    counts=frame.groupby('host')[features[0]].transform('size').to_numpy()
    xa=X.to_numpy(float); sa=sums.to_numpy(float)
    arr=np.zeros_like(xa); mask=counts>1
    arr[mask]=xa[mask]-((sa[mask]-xa[mask])/(counts[mask,None]-1))
    return pd.DataFrame(arr,columns=['hc_'+c for c in features])

def loo_cond(frame,features):
    f=frame.reset_index(drop=True)
    return pd.concat([f[features].astype(float),loo_center(f,features)],axis=1)

def stable_int(text):
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:8],16)

def subset_list(N,n,rng,max_subsets=2000):
    if n==N: return [np.arange(N,dtype=int)]
    total=math.comb(N,n)
    if total<=max_subsets:
        return [np.fromiter(c,dtype=int) for c in itertools.combinations(range(N),n)]
    seen=set(); out=[]
    while len(out)<max_subsets:
        a=tuple(sorted(rng.choice(N,size=n,replace=False).tolist()))
        if a not in seen:
            seen.add(a); out.append(np.array(a,dtype=int))
    return out

def pair_counts(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]
    if not len(pos) or not len(neg): return 0.0,0
    d=pos[:,None]-neg[None,:]
    return float((d>0).sum()+0.5*(d==0).sum()), int(d.size)

D=pd.read_csv(ROOT/'inputs/development_representation.csv')
T=pd.read_csv(ROOT/'inputs/development_targets_manifest.csv')[['record_id','authoritative_json_magmom']]
T['abs_moment']=T.authoritative_json_magmom.abs()
D=D.merge(T[['record_id','abs_moment']],on='record_id',validate='one_to_one')
features=sorted([c for c in D if c.startswith('x_')]+[c for c in D if c.startswith('c_') and 'delta' in c])
dev=D[['record_id','host_key','target_is_magnetic','abs_moment',*features]].rename(columns={'host_key':'host','target_is_magnetic':'y'})
base=pd.read_csv(ROOT/'outputs/tables/DEVELOPMENT_CROSSFIT_PREDICTIONS.csv')[['record_id','host','ET_score']]
base_map=base.set_index('record_id').ET_score.to_dict()
pool_sizes=[2,3,5,10,'full']
agg={p:{'cond_conc':0.0,'pairs':0,'et_conc':0.0,'subsets':0,'hosts':set()} for p in pool_sizes}
hostrows=[]; fullpred=[]
allhosts=sorted(dev.host.unique())
mixedhosts=[h for h in allhosts if dev.loc[dev.host.eq(h),'y'].nunique()==2]
for hi,h in enumerate(mixedhosts,1):
    train=dev[dev.host!=h].copy().reset_index(drop=True)
    test=dev[dev.host==h].copy().reset_index(drop=True)
    Xtr=loo_cond(train,features); w=rr.host_class_weights(train)
    off=stable_int(h); models=[]
    for seed in cfg['seeds']:
        m=rr.build_conditional_et(int(seed)+off+303,cfg)
        m.fit(Xtr,train.y,sample_weight=w); models.append(m)
    if test.y.nunique()==2:
        for p in pool_sizes:
            n=len(test) if p=='full' else int(p)
            if len(test)<n: continue
            rng=np.random.default_rng(20261002+off+n)
            subs=subset_list(len(test),n,rng,max_subsets=1000)
            raw=test[features].to_numpy(float); chunks=[]
            for idx in subs:
                Xs=raw[idx]
                other_mean=(Xs.sum(axis=0,keepdims=True)-Xs)/(n-1)
                chunks.append(np.concatenate([Xs,Xs-other_mean],axis=1))
            Xall=np.vstack(chunks)
            Xall=pd.DataFrame(Xall,columns=Xtr.columns); probs=np.mean([m.predict_proba(Xall)[:,1] for m in models],axis=0)
            cursor=0; hcc=0.; hpairs=0; het=0.
            for idx in subs:
                k=len(idx); pr=probs[cursor:cursor+k]; cursor+=k
                yy=test.iloc[idx].y.to_numpy(int)
                ets=np.array([base_map[r] for r in test.iloc[idx].record_id])
                c,pairs=pair_counts(yy,pr); e,_=pair_counts(yy,ets)
                hcc+=c; het+=e; hpairs+=pairs
            if hpairs:
                a=agg[p]; a['cond_conc']+=hcc; a['et_conc']+=het; a['pairs']+=hpairs; a['subsets']+=len(subs); a['hosts'].add(h)
                hostrows.append({'host':h,'pool_size':p,'n_records':len(test),'n_subsets':len(subs),'pairs':hpairs,'loo_cond_auc':hcc/hpairs,'et_auc':het/hpairs,'delta':(hcc-het)/hpairs})
            if p=='full':
                for j,row in test.iterrows():
                    fullpred.append({'record_id':row.record_id,'host':h,'y':int(row.y),'loo_cond_score':float(probs[j]),'ET_score':float(base_map[row.record_id])})
    print(f'{hi:02d}/{len(mixedhosts)} {h} n={len(test)} mixed={test.y.nunique()==2}',flush=True)
summary=[]
for p in pool_sizes:
    a=agg[p]
    if a['pairs']:
        summary.append({'pool_size':p,'evaluable_hosts':len(a['hosts']),'subsets':a['subsets'],'pair_comparisons':a['pairs'],'loo_cond_within_host_auc':a['cond_conc']/a['pairs'],'et_within_host_auc_same_subsets':a['et_conc']/a['pairs'],'delta':(a['cond_conc']-a['et_conc'])/a['pairs']})
S=pd.DataFrame(summary); H=pd.DataFrame(hostrows); P=pd.DataFrame(fullpred)
out=OUT; out.mkdir(exist_ok=True)
S.to_csv(out/'LOO_POOL_SENSITIVITY_SUMMARY.csv',index=False)
H.to_csv(out/'LOO_POOL_SENSITIVITY_BY_HOST.csv',index=False)
P.to_csv(out/'LOO_FULLPOOL_OOF_PREDICTIONS_MIXED_HOSTS.csv',index=False)
# Deterministic host-block bootstrap CIs across evaluable hosts.
rng_boot=np.random.default_rng(20261002)
cirows=[]
for p in pool_sizes:
    hh=H[H.pool_size.astype(str).eq(str(p))].copy()
    if hh.empty:
        continue
    vals=[]
    for _ in range(5000):
        sb=hh.iloc[rng_boot.integers(0,len(hh),len(hh))]
        ww=sb['pairs'].to_numpy(float)
        ca=float(np.sum(sb['loo_cond_auc'].to_numpy(float)*ww)/np.sum(ww))
        ea=float(np.sum(sb['et_auc'].to_numpy(float)*ww)/np.sum(ww))
        vals.append((ca,ea,ca-ea))
    vals=np.asarray(vals,float)
    q=np.quantile(vals,[0.025,0.975],axis=0)
    ww=hh['pairs'].to_numpy(float)
    ca=float(np.sum(hh['loo_cond_auc'].to_numpy(float)*ww)/np.sum(ww))
    ea=float(np.sum(hh['et_auc'].to_numpy(float)*ww)/np.sum(ww))
    cirows.append({'pool_size':p,'evaluable_hosts':len(hh),'cond_auc':ca,'cond_ci_low':q[0,0],'cond_ci_high':q[1,0],
                   'et_auc':ea,'et_ci_low':q[0,1],'et_ci_high':q[1,1],
                   'delta':ca-ea,'delta_ci_low':q[0,2],'delta_ci_high':q[1,2]})
pd.DataFrame(cirows).to_csv(out/'LOO_POOL_SENSITIVITY_BOOTSTRAP_CI.csv',index=False)

print('\nSUMMARY'); print(S.to_string(index=False))
print('\nFULL HOST ROWS'); print(H[H.pool_size.astype(str).eq('full')].to_string(index=False))
