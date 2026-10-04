from pathlib import Path
import sys, hashlib
import numpy as np, pandas as pd, yaml
PKG=Path(__file__).resolve().parents[1]
ROOT=PKG/'base_project'
OUT=PKG/'revision_results'
sys.path.insert(0,str(ROOT/'scripts'))
import run_transport_rank as rr
cfg=yaml.safe_load((ROOT/'config/project_config.yaml').read_text())

def stable_int(text): return int(hashlib.sha256(str(text).encode()).hexdigest()[:8],16)

def pair_counts(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); pos=s[y==1]; neg=s[y==0]
    if not len(pos) or not len(neg): return 0.,0
    d=pos[:,None]-neg[None,:]
    return float((d>0).sum()+.5*(d==0).sum()), int(d.size)
D=pd.read_csv(ROOT/'inputs/development_representation.csv')
T=pd.read_csv(ROOT/'inputs/development_targets_manifest.csv')[['record_id','authoritative_json_magmom']]; T['abs_moment']=T.authoritative_json_magmom.abs()
D=D.merge(T[['record_id','abs_moment']],on='record_id')
features=sorted([c for c in D if c.startswith('x_')]+[c for c in D if c.startswith('c_') and 'delta' in c])
dev=D[['record_id','host_key','target_is_magnetic','abs_moment',*features]].rename(columns={'host_key':'host','target_is_magnetic':'y'})
# dimensionality
Xc=rr.conditional_features(dev,features)
arr=Xc.to_numpy(float); std=arr.std(0,ddof=0); non=std>1e-12
Z=(arr[:,non]-arr[:,non].mean(0))/std[non]
s=np.linalg.svd(Z,compute_uv=False); eig=(s**2)/(len(Z)-1); pr=(eig.sum()**2)/(eig@eig); cum=np.cumsum(eig)/eig.sum()
rank=np.linalg.matrix_rank(Z,tol=1e-10); pc95=int(np.searchsorted(cum,.95)+1); pc99=int(np.searchsorted(cum,.99)+1)
raw=dev[features].to_numpy(float); cen=rr.centered_features(dev,features).to_numpy(float)
diag_corr=[]
for i in range(len(features)):
    if raw[:,i].std()>1e-12 and cen[:,i].std()>1e-12: diag_corr.append(np.corrcoef(raw[:,i],cen[:,i])[0,1])
# full model split share
w=rr.host_class_weights(dev); Xfull=rr.conditional_features(dev,features); models=[]
for seed in cfg['seeds']:
    m=rr.build_conditional_et(int(seed)+303,cfg); m.fit(Xfull,dev.y,sample_weight=w); models.append(m)
rawspl=0; censpl=0
for m in models:
  for tr in m.estimators_:
    f=tr.tree_.feature; f=f[f>=0]; rawspl += int((f<68).sum()); censpl += int((f>=68).sum())
# held-out-host permutation on original conditional model, mixed hosts only
mixed=[h for h,g in dev.groupby('host') if g.y.nunique()==2]
R=5
base_conc=0.; base_pairs=0
raw_conc=np.zeros(68); cen_conc=np.zeros(68); raw_pairs=np.zeros(68,int); cen_pairs=np.zeros(68,int)
block_raw_conc=block_cen_conc=0.; block_pairs=0
for hi,h in enumerate(mixed,1):
    trn=dev[dev.host!=h].reset_index(drop=True); tst=dev[dev.host==h].reset_index(drop=True)
    Xtr=rr.conditional_features(trn,features); Xte=rr.conditional_features(tst,features)
    weights=rr.host_class_weights(trn); off=stable_int(h); mods=[]
    for seed in cfg['seeds']:
        m=rr.build_conditional_et(int(seed)+off+303,cfg); m.fit(Xtr,trn.y,sample_weight=weights); mods.append(m)
    bp=np.mean([m.predict_proba(Xte)[:,1] for m in mods],axis=0); c,p=pair_counts(tst.y,bp); base_conc+=c; base_pairs+=p
    for r in range(R):
        rng=np.random.default_rng(20261002+off+1000*r)
        # same permutation for paired raw/centered counterpart
        perms=[rng.permutation(len(tst)) for _ in range(68)]
        # stack all 136 single-feature variants + 2 block variants
        mats=[]
        for j in range(68):
            a=Xte.to_numpy(float).copy(); a[:,j]=a[perms[j],j]; mats.append(a)
        for j in range(68):
            a=Xte.to_numpy(float).copy(); a[:,68+j]=a[perms[j],68+j]; mats.append(a)
        pb=rng.permutation(len(tst))
        a=Xte.to_numpy(float).copy(); a[:,:68]=a[pb,:68]; mats.append(a)
        a=Xte.to_numpy(float).copy(); a[:,68:]=a[pb,68:]; mats.append(a)
        big=pd.DataFrame(np.vstack(mats),columns=Xte.columns)
        pred=np.mean([m.predict_proba(big)[:,1] for m in mods],axis=0)
        n=len(tst); k=0
        for j in range(68):
            cc,pp=pair_counts(tst.y,pred[k*n:(k+1)*n]); raw_conc[j]+=cc; raw_pairs[j]+=pp; k+=1
        for j in range(68):
            cc,pp=pair_counts(tst.y,pred[k*n:(k+1)*n]); cen_conc[j]+=cc; cen_pairs[j]+=pp; k+=1
        cc,pp=pair_counts(tst.y,pred[k*n:(k+1)*n]); block_raw_conc+=cc; block_pairs+=pp; k+=1
        cc,pp=pair_counts(tst.y,pred[k*n:(k+1)*n]); block_cen_conc+=cc; k+=1
    print(f'{hi}/{len(mixed)} {h}',flush=True)
base_auc=base_conc/base_pairs
# since permutation repeated R times, baseline comparable repeated R
feature_rows=[]
for j,f in enumerate(features):
    rau=raw_conc[j]/raw_pairs[j]; cau=cen_conc[j]/cen_pairs[j]
    feature_rows.append({'descriptor':f,'raw_permuted_auc':rau,'raw_auc_drop':base_auc-rau,'centered_permuted_auc':cau,'centered_auc_drop':base_auc-cau})
F=pd.DataFrame(feature_rows)
out=OUT;out.mkdir(exist_ok=True)
F.sort_values('centered_auc_drop',ascending=False).to_csv(out/'MECHANISM_FEATURE_PERMUTATION.csv',index=False)
summary={
'nominal_variables':136,'nonconstant_variables':int(non.sum()),'standardized_rank':int(rank),'participation_ratio':float(pr),'pcs_95pct':pc95,'pcs_99pct':pc99,
'raw_centered_counterpart_corr_mean':float(np.mean(diag_corr)),'raw_centered_counterpart_corr_median':float(np.median(diag_corr)),'raw_centered_counterpart_corr_min':float(np.min(diag_corr)),'raw_centered_counterpart_corr_max':float(np.max(diag_corr)),
'raw_splits':rawspl,'centered_splits':censpl,'total_splits':rawspl+censpl,'centered_split_share':censpl/(rawspl+censpl),
'permutation_base_within_host_auc':base_auc,
'raw_block_permuted_auc':block_raw_conc/block_pairs,'raw_block_auc_drop':base_auc-block_raw_conc/block_pairs,
'centered_block_permuted_auc':block_cen_conc/block_pairs,'centered_block_auc_drop':base_auc-block_cen_conc/block_pairs,
'permutation_repeats':R,'evaluable_hosts':len(mixed),'within_host_pairs':base_pairs
}
import json
(out/'MECHANISM_DIMENSIONALITY_SPLITS_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print('\nSUMMARY'); print(json.dumps(summary,indent=2))
print('\nTOP CENTERED'); print(F.sort_values('centered_auc_drop',ascending=False).head(12).to_string(index=False))
print('\nTOP RAW'); print(F.sort_values('raw_auc_drop',ascending=False).head(12).to_string(index=False))
