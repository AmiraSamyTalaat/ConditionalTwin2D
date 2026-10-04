from __future__ import annotations
import argparse
import json
from pathlib import Path

REQUIRED = [
    "STATUS.json",
    "SUCCESS_MARKER.json",
    "OUTPUT_VERIFICATION.json",
    "GRAPH_DESCRIPTOR_TABLE.csv",
    "GRAPH_EXTRACTION_ERRORS.csv",
    "GRAPH_EXTRACTION_AUDIT.json",
    "FEATURE_SET_DEFINITIONS.csv",
    "LOHO_BINARY_PREDICTIONS.csv",
    "LOHO_BINARY_SUMMARY.csv",
    "LOHO_BINARY_HOST_METRICS.csv",
    "LOHO_ORDINAL_PREDICTIONS.csv",
    "LOHO_ORDINAL_SUMMARY.csv",
    "LOHO_ORDINAL_HOST_METRICS.csv",
    "LEAKAGE_AUDIT.json",
    "STAGE2B_GATE_DECISION.json",
    "ARABIC_STAGE2B_SUMMARY.md",
    "FEATURE_AND_LEAKAGE_RULES.md",
    "STAGE2B_GATE_RULES.md",
    "RUN_LOG.txt",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()
    out = Path(args.output)

    missing = [
        name
        for name in REQUIRED
        if not (out / name).exists()
        or (out / name).stat().st_size == 0
    ]
    if not (
        out / "ERROR_LOG.txt"
    ).exists():
        missing.append(
            "ERROR_LOG.txt (may be empty)"
        )
    if missing:
        print(
            "MISSING/EMPTY:",
            *missing,
            sep="\n- ",
        )
        return 2

    status = json.loads(
        (
            out / "STATUS.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    marker = json.loads(
        (
            out / "SUCCESS_MARKER.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (
            out
            / "STAGE2B_GATE_DECISION.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    verification = json.loads(
        (
            out
            / "OUTPUT_VERIFICATION.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if status.get("state") != "COMPLETED":
        print(
            "STATUS is not COMPLETED."
        )
        return 3
    if not marker.get(
        "output_verification_passed"
    ):
        print(
            "Output verification failed."
        )
        return 4
    if not verification.get(
        "all_required_present"
    ):
        print(
            "Required output verification failed."
        )
        return 5

    print("OUTPUT VERIFICATION PASSED")
    print(
        "Stage 2B decision:",
        decision["stage2b_decision"],
    )
    print(
        "Metadata binary AUROC:",
        decision[
            "metadata_binary_AUROC"
        ],
    )
    print(
        "Best structural binary set:",
        decision[
            "best_structural_binary_feature_set"
        ],
    )
    print(
        "Best structural binary AUROC:",
        decision[
            "best_structural_binary_AUROC"
        ],
    )
    print(
        "Metadata ordinal QWK:",
        decision[
            "metadata_ordinal_QWK"
        ],
    )
    print(
        "Best structural ordinal set:",
        decision[
            "best_structural_ordinal_feature_set"
        ],
    )
    print(
        "Best structural ordinal QWK:",
        decision[
            "best_structural_ordinal_QWK"
        ],
    )
    print(
        "Next allowed stage:",
        decision["next_allowed_stage"],
    )
    print(
        "Actual GNN pilot allowed:",
        decision[
            "actual_gnn_training_allowed"
        ],
    )
    print(
        "Final CDT-EGNN training allowed:",
        False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
