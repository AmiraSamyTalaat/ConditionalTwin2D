from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
required = [
    OUT / "SUCCESS_MARKER.json",
    OUT / "VERIFICATION.json",
    OUT / "Q1_STRENGTHENING_REPORT.md",
    OUT / "ARABIC_EXECUTIVE_SUMMARY.md",
    OUT / "QUANTITATIVE_AD_LOCK.json",
    OUT / "tables" / "TABLE_Q1_CALIBRATION_METHOD_COMPARISON.csv",
    OUT / "tables" / "TABLE_Q1_ENDPOINT_SENSITIVITY.csv",
    OUT / "tables" / "TABLE_Q1_CONTINUOUS_TARGET.csv",
    OUT / "tables" / "TABLE_Q1_BETWEEN_WITHIN_DECOMPOSITION.csv",
    OUT / "tables" / "TABLE_Q1_AD_COHORT_SUMMARY.csv",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists() or p.stat().st_size == 0]
verification = json.loads((OUT / "VERIFICATION.json").read_text(encoding="utf-8")) if (OUT / "VERIFICATION.json").exists() else {}
checks = {
    "required_files_present": not missing,
    "analysis_verification_all": verification.get("all") is True,
    "calibration_rows_4": len(pd.read_csv(OUT / "tables" / "TABLE_Q1_CALIBRATION_METHOD_COMPARISON.csv")) == 4 if not missing else False,
    "endpoint_rows_12": len(pd.read_csv(OUT / "tables" / "TABLE_Q1_ENDPOINT_SENSITIVITY.csv")) == 12 if not missing else False,
    "figure_sets_8": all(len(list((OUT / "figures").glob(f"*.{ext}"))) >= 8 for ext in ["png", "pdf", "svg"]),
    "missing": missing,
}
checks["all"] = all(v for k, v in checks.items() if k != "missing")
(OUT / "FINAL_VERIFICATION.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
if not checks["all"]:
    raise SystemExit(1)
print("Verification PASS")
