from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
required=[
"tables/DefectTwin2D_Final_Q1_Repair_Tables.xlsx",
"tables/TABLE_FINAL_CALIBRATION_COMPARISON.csv",
"tables/TABLE_FINAL_THRESHOLD_CONFIDENCE_INTERVALS.csv",
"tables/TABLE_FINAL_CONTINUOUS_TARGET.csv",
"tables/TABLE_FINAL_EXPLICIT_BASELINES.csv",
"tables/TABLE_FINAL_AD_V2_COHORT_SUMMARY.csv",
"analysis/REVIEWER_COMPLETENESS_MATRIX.csv",
"analysis/FINAL_Q1_COMPLETION_REPORT.md",
"analysis/FINAL_SCIENTIFIC_SUMMARY.json",
]
missing=[x for x in required if not (ROOT/x).exists() or (ROOT/x).stat().st_size==0]
if missing: raise SystemExit("Missing or empty scientific artifact(s): "+str(missing))
cal=pd.read_csv(ROOT/"tables/TABLE_FINAL_CALIBRATION_COMPARISON.csv")
sel=cal[cal.selected_on_development_only.astype(bool)]
assert len(sel)==1 and sel.method.iloc[0]=="temperature"
review=pd.read_csv(ROOT/"analysis/REVIEWER_COMPLETENESS_MATRIX.csv")
assert len(review)>=20
summary=json.loads((ROOT/"analysis/FINAL_SCIENTIFIC_SUMMARY.json").read_text())
assert summary["external_tuning_allowed"] is False
assert summary["final_CDT_EGNN_allowed"] is False
print("BASE DEFECTTWIN SCIENTIFIC PACKAGE VERIFICATION PASSED")
print("Selected calibrator: temperature")
print("Reviewer matrix rows:",len(review))
print("External tuning allowed: False")
print("Final CDT-EGNN allowed: False")
