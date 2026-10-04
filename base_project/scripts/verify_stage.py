from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


required = [
    OUT / "SUCCESS.json",
    OUT / "STATUS.json",
    OUT / "VERIFICATION.json",
    OUT / "FINAL_DEVELOPMENT_SELECTED_CONDITIONALTWIN.json",
    OUT / "HASHES_SHA256.csv",
    OUT / "RESULT_PACKAGE.zip",
    OUT / "tables" / "TABLE_MODEL_COHORT_SUMMARY.csv",
    OUT / "tables" / "TABLE_PAIRED_BOOTSTRAP_DIFFERENCES.csv",
    OUT / "tables" / "DEVELOPMENT_ROW_EXCLUSION_SELECTION_SENSITIVITY.csv",
    OUT / "tables" / "DefectTwin2D_TransportRank_Tables.xlsx",
]
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing or empty: {path.relative_to(ROOT)}")

status = json.loads((OUT / "STATUS.json").read_text(encoding="utf-8"))
verification = json.loads((OUT / "VERIFICATION.json").read_text(encoding="utf-8"))
selection = json.loads((OUT / "FINAL_DEVELOPMENT_SELECTED_CONDITIONALTWIN.json").read_text(encoding="utf-8"))
if status.get("state") != "SUCCESS":
    fail("STATUS.json is not SUCCESS")
if verification.get("all") is not True:
    fail("VERIFICATION.json all != true")
if selection.get("name") != "Conditional_only":
    fail("Unexpected selected branch")
if selection.get("external_labels_used") is not False:
    fail("External labels were marked as used")

with (OUT / "tables" / "DEVELOPMENT_ROW_EXCLUSION_SELECTION_SENSITIVITY.csv").open(encoding="utf-8-sig", newline="") as f:
    folds = list(csv.DictReader(f))
if len(folds) != 30 or any(row.get("name") != "Conditional_only" for row in folds):
    fail("Conditional_only was not retained in all 30 host-row-exclusion sensitivity tables")

with (OUT / "tables" / "TABLE_PAIRED_BOOTSTRAP_DIFFERENCES.csv").open(encoding="utf-8-sig", newline="") as f:
    paired = list(csv.DictReader(f))
match = [r for r in paired if r["cohort"] == "Development_LOHO" and r["comparison"] == "ConditionalET - ET" and r["metric"] == "within_host_pair_AUROC"]
if len(match) != 1:
    fail("Missing paired development within-host comparison")
row = match[0]
if float(row["ci95_low"]) <= 0 or float(row["mean_difference"]) <= 0:
    fail("Paired within-host improvement is not positive with positive lower CI")

manifest_path = OUT / "HASHES_SHA256.csv"
with manifest_path.open(encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))
for row in manifest:
    path = ROOT / row["relative_path"]
    if not path.is_file():
        fail(f"Manifest file missing: {row['relative_path']}")
    if sha256(path) != row["sha256"]:
        fail(f"Hash mismatch: {row['relative_path']}")

with zipfile.ZipFile(OUT / "RESULT_PACKAGE.zip") as z:
    bad = z.testzip()
    if bad:
        fail(f"Corrupt member in RESULT_PACKAGE.zip: {bad}")

print("VERIFICATION PASSED")
print("Selected method: ConditionalTwin-2D / Conditional_only")
print("Development within-host improvement has positive paired 95% CI")
print("External cohorts remain retrospective diagnostics")
