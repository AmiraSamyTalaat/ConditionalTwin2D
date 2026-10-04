from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
pkg=Path(__file__).resolve().parents[1]
root=pkg/'base_project'
out=pkg/'revision_results'; out.mkdir(exist_ok=True)
D=pd.read_csv(root/'inputs/development_representation.csv')
T=pd.read_csv(root/'inputs/development_targets_manifest.csv')[['record_id','authoritative_json_magmom']]
D=D.merge(T,on='record_id')
features=sorted([c for c in D if c.startswith('x_')]+[c for c in D if c.startswith('c_') and 'delta' in c])
X=D[features].astype(float)
non=X.std(0)>1e-12; X=X.loc[:,non]
Z=StandardScaler().fit_transform(X)
pca=PCA(n_components=10,random_state=0).fit(Z); P=pca.transform(Z)
res=pd.DataFrame({'record_id':D.record_id,'host':D.host_key,'y':D.target_is_magnetic,'abs_moment':D.authoritative_json_magmom.abs(),'PC1':P[:,0],'PC2':P[:,1]})
res.to_csv(out/'PCA_DEVELOPMENT_COORDINATES.csv',index=False)
summary={'pc1_variance':float(pca.explained_variance_ratio_[0]),'pc2_variance':float(pca.explained_variance_ratio_[1]),'pc12_variance':float(pca.explained_variance_ratio_[:2].sum()),'pcs_10_variance':float(pca.explained_variance_ratio_.sum())}
# silhouette on PC1/2 and full standardized for binary class; host only where >1 record
try: summary['binary_silhouette_pc12']=float(silhouette_score(P[:,:2],D.target_is_magnetic))
except: pass
try: summary['binary_silhouette_full68']=float(silhouette_score(Z,D.target_is_magnetic))
except: pass
# moment plot
fig,ax=plt.subplots(figsize=(7,5.5))
sc=ax.scatter(P[:,0],P[:,1],c=np.log1p(D.authoritative_json_magmom.abs()),s=22,alpha=.75)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Development descriptors colored by log(1 + |magnetic moment|)')
cb=fig.colorbar(sc,ax=ax); cb.set_label('log(1 + |M|)')
fig.tight_layout(); fig.savefig(out/'PCA_BY_MAGNETIC_MOMENT.png',dpi=300); fig.savefig(out/'PCA_BY_MAGNETIC_MOMENT.pdf'); plt.close(fig)
# host plot categorical numeric with centroid labels
hosts=sorted(D.host_key.unique()); codes=pd.Categorical(D.host_key,categories=hosts).codes
fig,ax=plt.subplots(figsize=(8,6.5))
sc=ax.scatter(P[:,0],P[:,1],c=codes,s=20,alpha=.7)
for h in hosts:
 m=(D.host_key==h).to_numpy();
 if m.sum()>=2:
  ax.text(P[m,0].mean(),P[m,1].mean(),h,fontsize=6)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Development descriptors colored by host with host centroids labeled')
fig.tight_layout(); fig.savefig(out/'PCA_BY_HOST.png',dpi=300); fig.savefig(out/'PCA_BY_HOST.pdf'); plt.close(fig)
import json
(out/'PCA_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(summary)
