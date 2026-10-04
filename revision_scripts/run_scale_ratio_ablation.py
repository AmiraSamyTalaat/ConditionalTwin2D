from pathlib import Path
import sys,hashlib
import pandas as pd,numpy as np,yaml
PKG=Path(__file__).resolve().parents[1];ROOT=PKG/'base_project';OUT=PKG/'revision_results';sys.path.insert(0,str(ROOT/'scripts'))
import run_transport_rank as rr
cfg=yaml.safe_load((ROOT/'config/project_config.yaml').read_text())
def stable_int(x): return int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)
def pc(y,s):
 y=np.asarray(y);s=np.asarray(s);p=s[y==1];n=s[y==0]
 if not len(p) or not len(n): return 0.,0
 d=p[:,None]-n[None,:];return float((d>0).sum()+.5*(d==0).sum()),int(d.size)
D=pd.read_csv(ROOT/'inputs/development_representation.csv');features=sorted([c for c in D if c.startswith('x_')]+[c for c in D if c.startswith('c_') and 'delta' in c]);dev=D[['record_id','host_key','target_is_magnetic',*features]].rename(columns={'host_key':'host','target_is_magnetic':'y'})
sets={'full68':features,'drop_atom_area':[f for f in features if f not in ['x_atom_count_ratio','x_area_ratio']]}
rows=[]
for name,fs in sets.items():
 conc=0.;pairs=0
 for h,g in dev.groupby('host'):
  if g.y.nunique()!=2: continue
  tr=dev[dev.host!=h].reset_index(drop=True);te=dev[dev.host==h].reset_index(drop=True);off=stable_int(h);w=rr.host_class_weights(tr)
  Xtr=rr.conditional_features(tr,fs);Xte=rr.conditional_features(te,fs);mods=[]
  for seed in cfg['seeds']:
   m=rr.build_conditional_et(int(seed)+off+303,cfg);m.fit(Xtr,tr.y,sample_weight=w);mods.append(m)
  p=np.mean([m.predict_proba(Xte)[:,1] for m in mods],axis=0);c,n=pc(te.y,p);conc+=c;pairs+=n
 print(name,conc/pairs,pairs,flush=True);rows.append({'feature_set':name,'within_host_auc':conc/pairs,'pairs':pairs,'n_raw_features':len(fs)})
OUT.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(OUT/'SCALE_RATIO_ABLATION.csv',index=False)
