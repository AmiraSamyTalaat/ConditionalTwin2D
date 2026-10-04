from pathlib import Path
import sys, hashlib, json, time, warnings
import numpy as np, pandas as pd, yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[1]; INPUT=ROOT/'inputs'; OUT=ROOT/'final_outputs'; CHECK=OUT/'checkpoints'; TABLE=OUT/'tables'; CHECK.mkdir(parents=True,exist_ok=True); TABLE.mkdir(parents=True,exist_ok=True)
seed=int(sys.argv[1])
def stable_int(text): return int(hashlib.sha256(str(text).encode()).hexdigest()[:8],16)
def weights(hosts):
    c=hosts.value_counts(); w=hosts.map(lambda h:1/c[h]).to_numpy(float); return w/w.mean()
def fit(name,y,p,w):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-7,1-1e-7);w=np.asarray(w,float);z=logit(p)
    if name=='identity': return (),True
    if name=='platt':
        def f(t):
            a=np.exp(t[0]);c=t[1];q=expit(a*z+c);return np.average(-(y*np.log(q+1e-12)+(1-y)*np.log(1-q+1e-12)),weights=w)
        r=minimize(f,[0,0],method='L-BFGS-B');return (float(np.exp(r.x[0])),float(r.x[1])),bool(r.success)
    if name=='beta':
        lp=np.log(p);lq=-np.log1p(-p)
        def f(t):
            a=np.exp(t[0]);b=np.exp(t[1]);c=t[2];q=expit(a*lp+b*lq+c);return np.average(-(y*np.log(q+1e-12)+(1-y)*np.log(1-q+1e-12)),weights=w)
        r=minimize(f,[0,0,0],method='L-BFGS-B');return (float(np.exp(r.x[0])),float(np.exp(r.x[1])),float(r.x[2])),bool(r.success)
    if name=='temperature':
        def f(t):
            T=np.exp(t[0]);q=expit(z/T);return np.average(-(y*np.log(q+1e-12)+(1-y)*np.log(1-q+1e-12)),weights=w)
        r=minimize(f,[0],method='L-BFGS-B');return (float(np.exp(r.x[0])),),bool(r.success)
def apply(name,params,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7);z=logit(p)
    if name=='identity': return p
    if name=='platt':a,c=params;return expit(a*z+c)
    if name=='beta':a,b,c=params;return expit(a*np.log(p)+b*(-np.log1p(-p))+c)
    if name=='temperature':return expit(z/params[0])
cfg=yaml.safe_load((ROOT/'config/project_config.yaml').read_text())
dev=pd.read_csv(INPUT/'development_representation.csv');t=pd.read_csv(INPUT/'development_targets_manifest.csv')[['record_id','authoritative_json_magmom']];t['abs_moment']=t.authoritative_json_magmom.abs();dev=dev.merge(t[['record_id','abs_moment']],on='record_id')
base=pd.read_csv(INPUT/'original_base_loho_predictions.csv');base=base[base.model.eq('et_x_composition_delta_primary')]
features=sorted([c for c in dev if c.startswith('x_')]+[c for c in dev if c.startswith('c_') and 'delta' in c]);hosts=sorted(dev.host_key.unique());methods=['identity','platt','beta','temperature']
rows=[];pars=[];start=time.time()
for hi,test_host in enumerate(hosts,1):
    train=dev[dev.host_key.ne(test_host)].reset_index(drop=True);test=dev[dev.host_key.eq(test_host)].copy();oof=np.full(len(train),np.nan)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed+stable_int(test_host))
    for fold,(tr,ca) in enumerate(splitter.split(train[features],train.target_is_magnetic,train.host_key)):
        a=train.iloc[tr];b=train.iloc[ca];model=ExtraTreesClassifier(n_estimators=220,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=(seed+stable_int(test_host)+fold*1009)%4294967295,n_jobs=-1)
        model.fit(a[features],a.target_is_magnetic,sample_weight=weights(a.host_key));oof[ca]=model.predict_proba(b[features])[:,1]
    outer=base[(base.seed.eq(seed))&(base.test_host.eq(test_host))].set_index('record_id');testp=test.record_id.map(outer.probability).to_numpy(float);w=weights(train.host_key)
    for m in methods:
        params,valid=fit(m,train.target_is_magnetic,oof,w)
        if not valid: raise RuntimeError(f'calibrator failed {m} {test_host}')
        pc=apply(m,params,testp);pars.append({'seed':seed,'test_host':test_host,'method':m,'params':json.dumps(params),'valid':valid})
        for rec,y,pu,pcal in zip(test.record_id,test.target_is_magnetic,testp,pc):rows.append({'seed':seed,'test_host':test_host,'record_id':rec,'y_true':int(y),'method':m,'uncalibrated_probability':float(pu),'calibrated_probability':float(pcal),'calibrator_valid':True})
    print(f'{seed} host {hi}/30 elapsed={time.time()-start:.1f}',flush=True)
pd.DataFrame(rows).to_csv(CHECK/f'calibration_seed_{seed}.csv',index=False);pd.DataFrame(pars).to_csv(CHECK/f'calibration_params_seed_{seed}.csv',index=False)
print('DONE',seed,len(rows),flush=True)
