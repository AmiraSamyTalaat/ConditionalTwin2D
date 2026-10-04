from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inputs" / "STAGE2A_FIX1_RESULTS.zip"
assert SOURCE.exists(), "STAGE2A_FIX1_RESULTS.zip is missing."

with zipfile.ZipFile(SOURCE) as archive:
    gate = json.loads(
        archive.read(
            "stage2a_fix1_output/STAGE2A_FIX1_GATE_DECISION.json"
        ).decode("utf-8")
    )
    manifest = pd.read_csv(io.BytesIO(
        archive.read(
            "stage2a_fix1_output/RECONCILED_GRAPH_READY_MANIFEST.csv"
        )
    ))
    defect_files = [
        name for name in archive.namelist()
        if name.startswith(
            "stage2a_fix1_output/structures/defect/"
        ) and name.endswith(".json")
    ]
    pristine_files = [
        name for name in archive.namelist()
        if name.startswith(
            "stage2a_fix1_output/structures/pristine/"
        ) and name.endswith(".json")
    ]

assert gate["stage2a_fix1_decision"] == "GREEN"
assert gate["safe_defect_files"] == 529
assert gate["safe_pristine_hosts"] == 30
assert len(manifest) == 529
assert manifest["pair_ready"].astype(bool).all()
assert len(defect_files) == 529
assert len(pristine_files) == 30

print("STAGE 2A FIX1 GREEN VERIFIED")
print("529 SAFE DEFECT STRUCTURES VERIFIED")
print("30 SAFE PRISTINE STRUCTURES VERIFIED")
print("GRAPH-DERIVED BASELINE INPUT READY")
