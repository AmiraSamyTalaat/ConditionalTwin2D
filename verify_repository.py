from pathlib import Path
import argparse, hashlib, subprocess, sys, tempfile, shutil, json, csv
ROOT=Path(__file__).resolve().parent

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def run(root, path):
    print(f"\n=== {path} ===", flush=True)
    subprocess.run([sys.executable, str(root/path)], cwd=root, check=True)

def verify_manifest(root=ROOT):
    manifest=root/'SHA256SUMS.txt'
    if not manifest.exists(): raise SystemExit('SHA256SUMS.txt missing')
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        digest, rel=line.split('  ',1)
        p=root/rel
        if not p.is_file(): raise SystemExit(f'Missing manifest file: {rel}')
        if sha256(p)!=digest: raise SystemExit(f'Hash mismatch: {rel}')
    print('ROOT SHA256 MANIFEST PASSED')

def verify_archived_slow_audits(root=ROOT):
    mech=json.loads((root/'revision_results/MECHANISM_DIMENSIONALITY_SPLITS_SUMMARY.json').read_text())
    checks=[
        (mech.get('standardized_rank')==86,'mechanism standardized rank'),
        (abs(mech.get('participation_ratio',0)-23.6633925995)<1e-6,'mechanism participation ratio'),
        (abs(mech.get('raw_centered_counterpart_corr_mean',0)-0.5769428696)<1e-6,'raw-centered mean correlation'),
        (abs(mech.get('centered_split_share',0)-0.4972556110)<1e-6,'centered split share'),
        (abs(mech.get('centered_block_auc_drop',0)-0.0504618543)<1e-6,'centered-block permutation drop'),
    ]
    for ok,label in checks:
        if not ok: raise SystemExit(f'Archived audit check failed: {label}')
    # Scale-ratio ablation archived result
    with (root/'revision_results/SCALE_RATIO_ABLATION.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise SystemExit('SCALE_RATIO_ABLATION.csv empty')
    nums=[]
    for r in rows:
        for k,v in r.items():
            try: nums.append(float(v))
            except Exception: pass
    if not any(abs(x-0.84767)<5e-4 for x in nums):
        raise SystemExit('Archived scale-ratio ablation value not found')
    # Pool-size deltas
    with (root/'revision_results/LOO_POOL_SENSITIVITY_SUMMARY.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    observed=[round(float(r['delta']),4) for r in rows]
    expected=[0.0753,0.0693,0.0524,0.0575,0.0582]
    if observed!=expected:
        raise SystemExit(f'Archived pool-size deltas unexpected: {observed}')
    print('ARCHIVED COMPUTATIONALLY EXPENSIVE AUDITS PASSED')

FAST_AUDITS=[
    'revision_scripts/run_metric_tie_audit.py',
    'revision_scripts/run_2dmd_host_audit.py',
    'revision_scripts/run_pca_audit.py',
    'revision_scripts/run_candidate_pool_completeness_audit.py',
    'revision_scripts/verify_graph_provenance.py',
]

ap=argparse.ArgumentParser()
ap.add_argument('--quick',action='store_true',help='Integrity and frozen-package checks only.')
ap.add_argument('--full-recompute',action='store_true',help='Additionally rerun every reviewer-facing audit in an isolated temporary copy. This can be computationally expensive.')
a=ap.parse_args()

# Check immutable archive integrity before computation.
verify_manifest(ROOT)
run(ROOT,'base_defecttwin_final_q1/scripts/verify_final_package.py')
run(ROOT,'base_project/scripts/verify_stage.py')

if not a.quick:
    # Default reviewer verification reruns the fast/medium audits in an isolated copy
    # and validates the archived outputs of the expensive mechanism/pool/ablation jobs.
    with tempfile.TemporaryDirectory(prefix='conditionaltwin_verify_') as td:
        tmp=Path(td)/'repo'
        shutil.copytree(ROOT,tmp,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        print(f"\n=== isolated reviewer audits ===\nTemporary working copy: {tmp}", flush=True)
        if a.full_recompute:
            run(tmp,'revision_scripts/run_all_revision_audits.py')
            print('ISOLATED FULL RECOMPUTATION PASSED')
        else:
            for script in FAST_AUDITS:
                run(tmp,script)
            print('ISOLATED FAST/MEDIUM AUDITS PASSED')
    verify_archived_slow_audits(ROOT)

# Confirm verification did not mutate the public archive.
verify_manifest(ROOT)
print('\nREPOSITORY VERIFICATION PASSED')
