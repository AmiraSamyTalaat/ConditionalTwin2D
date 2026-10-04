from pathlib import Path
import zipfile, io, importlib.util, json
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream_descriptor_generation' / 'STAGE2B_GRAPH_DERIVED_BASELINES'
RUN = UP / 'scripts' / 'run_stage2b.py'
CFG = yaml.safe_load((UP / 'config' / 'project_config.yaml').read_text(encoding='utf-8'))
STAGE2A = UP / 'inputs' / 'STAGE2A_FIX1_RESULTS.zip'
DEV = pd.read_csv(ROOT / 'base_project' / 'inputs' / 'development_representation.csv')

spec = importlib.util.spec_from_file_location('stage2b_original', RUN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

with zipfile.ZipFile(STAGE2A) as z:
    manifest = pd.read_csv(io.BytesIO(z.read('stage2a_fix1_output/RECONCILED_GRAPH_READY_MANIFEST.csv')))
    dreg = pd.read_csv(io.BytesIO(z.read('stage2a_fix1_output/SAFE_DEFECT_STRUCTURE_REGISTRY.csv')))
    preg = pd.read_csv(io.BytesIO(z.read('stage2a_fix1_output/SAFE_PRISTINE_STRUCTURE_REGISTRY.csv')))
    dmap = dict(zip(dreg['record_id'].astype(str), dreg['relative_path'].astype(str)))
    pmap = dict(zip(preg['host_key'].astype(str), preg['relative_path'].astype(str)))
    pfeat = {}
    for host, rel in pmap.items():
        s = mod.load_structure(z, 'stage2a_fix1_output/' + rel.replace('\\','/'))
        pfeat[str(host)] = mod.structure_features(s, CFG, 'p_')

    sampled = manifest.groupby('host_key', sort=True).head(1)
    rows=[]
    max_abs=0.0
    mismatches=0
    comparisons=0
    for r in sampled.itertuples(index=False):
        rid, host = str(r.record_id), str(r.host_key)
        ds = mod.load_structure(z, 'stage2a_fix1_output/' + dmap[rid].replace('\\','/'))
        df = mod.structure_features(ds, CFG, 'd_')
        pf = pfeat[host]
        x={}
        for dn in mod.intensive_feature_names(list(df), 'd_'):
            suf=dn[2:]
            pn='p_'+suf
            if pn in pf:
                x['x_'+suf]=df[dn]-pf[pn]
        x['x_atom_count_ratio']=df['d_n_atoms']/pf['p_n_atoms']
        x['x_area_ratio']=df['d_cell_area_A2']/pf['p_cell_area_A2']
        x['x_density_ratio']=df['d_atoms_per_area']/pf['p_atoms_per_area']
        target=DEV.loc[DEV['record_id'].astype(str).eq(rid)].iloc[0]
        local_max=0.0
        local_bad=0
        common=[c for c in x if c in DEV.columns]
        for c in common:
            a=float(x[c]); b=float(target[c])
            diff=0.0 if np.isnan(a) and np.isnan(b) else abs(a-b)
            comparisons += 1
            max_abs=max(max_abs,diff)
            local_max=max(local_max,diff)
            if diff > 1e-10:
                mismatches += 1
                local_bad += 1
        rows.append({'record_id':rid,'host_key':host,'features_compared':len(common),'max_abs_difference':local_max,'differences_gt_1e-10':local_bad})

outdir=ROOT/'revision_results'
pd.DataFrame(rows).to_csv(outdir/'GRAPH_PROVENANCE_AUDIT_BY_HOST.csv',index=False)
summary={
  'source_stage':'STAGE2B_GRAPH_DERIVED_BASELINE_COMPARISON',
  'hosts_sampled':int(len(rows)),
  'graph_features_compared_per_host':int(rows[0]['features_compared']) if rows else 0,
  'total_comparisons':int(comparisons),
  'maximum_absolute_difference':float(max_abs),
  'differences_above_1e-10':int(mismatches),
  'graph_config':CFG['graph'],
  'pass': bool(len(rows)==30 and comparisons==1560 and mismatches==0)
}
(outdir/'GRAPH_PROVENANCE_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
