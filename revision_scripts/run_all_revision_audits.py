from pathlib import Path
import subprocess, sys
HERE = Path(__file__).resolve().parent
scripts = [
    'run_metric_tie_audit.py',
    'run_2dmd_host_audit.py',
    'run_pca_audit.py',
    'run_candidate_pool_completeness_audit.py',
    'verify_graph_provenance.py',
    'run_scale_ratio_ablation.py',
    'run_loo_pool_sensitivity.py',
    'run_mechanism_audit.py',
]
for name in scripts:
    print(f'\n=== {name} ===', flush=True)
    subprocess.run([sys.executable, str(HERE/name)], check=True)
print('\nALL REVISION AUDITS PASSED', flush=True)
