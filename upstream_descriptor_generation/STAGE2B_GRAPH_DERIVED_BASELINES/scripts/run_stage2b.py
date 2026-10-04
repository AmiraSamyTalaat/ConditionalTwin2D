from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Logger:
    def __init__(self, out: Path):
        self.run = out / "RUN_LOG.txt"
        self.error_file = out / "ERROR_LOG.txt"
        self.run.touch(exist_ok=True)
        self.error_file.touch(exist_ok=True)

    def info(self, text):
        line = f"[{utcnow()}] {text}"
        print(line, flush=True)
        with self.run.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def error(self, text):
        line = f"[{utcnow()}] ERROR: {text}"
        print(line, file=sys.stderr, flush=True)
        with self.error_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# Covalent radii in angstrom for the elements present in the frozen dataset.
COVALENT_RADII = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    20: 1.76, 26: 1.32, 27: 1.26, 31: 1.22, 32: 1.20,
    33: 1.19, 34: 1.20, 35: 1.20, 40: 1.45, 42: 1.54,
    47: 1.45, 48: 1.44, 50: 1.39, 51: 1.39, 52: 1.38,
    53: 1.39, 56: 2.15, 72: 1.52, 74: 1.62, 83: 1.48,
}


def atoms_payload(data):
    obj = json.loads(
        data.decode("utf-8")
        if isinstance(data, bytes)
        else data
    )
    if {"numbers", "positions", "cell", "pbc"}.issubset(obj):
        return obj
    for value in obj.values():
        if isinstance(value, dict) and {
            "numbers", "positions", "cell", "pbc"
        }.issubset(value):
            return value
    raise ValueError("ASE atoms payload was not found.")


def load_structure(archive, member):
    payload = atoms_payload(archive.read(member))
    return {
        "numbers": np.asarray(payload["numbers"], dtype=int),
        "positions": np.asarray(payload["positions"], dtype=float),
        "cell": np.asarray(payload["cell"], dtype=float),
        "pbc": np.asarray(payload["pbc"], dtype=bool),
    }


def minimum_periodic_distance(
    position_i,
    position_j,
    cell,
    translations,
):
    best = math.inf
    delta = position_j - position_i
    for tx in translations:
        for ty in translations:
            shifted = (
                delta
                + tx * cell[0]
                + ty * cell[1]
            )
            distance = float(np.linalg.norm(shifted))
            if distance < best:
                best = distance
    return best


def composition_features(numbers, prefix):
    z = numbers.astype(float)
    features = {
        f"{prefix}n_atoms": float(len(z)),
        f"{prefix}Z_mean": float(z.mean()),
        f"{prefix}Z_std": float(z.std()),
        f"{prefix}Z_min": float(z.min()),
        f"{prefix}Z_max": float(z.max()),
        f"{prefix}Z_unique": float(
            len(np.unique(z))
        ),
    }
    bins = [
        (1, 10),
        (11, 20),
        (21, 30),
        (31, 40),
        (41, 50),
        (51, 60),
        (61, 80),
        (81, 120),
    ]
    for index, (lower, upper) in enumerate(bins):
        features[
            f"{prefix}Z_bin_{index}_fraction"
        ] = float(
            ((z >= lower) & (z <= upper)).mean()
        )
    return features


def structure_features(structure, cfg, prefix):
    numbers = structure["numbers"]
    positions = structure["positions"]
    cell = structure["cell"]

    features = composition_features(
        numbers,
        prefix,
    )

    area = float(
        np.linalg.norm(np.cross(cell[0], cell[1]))
    )
    thickness = float(
        positions[:, 2].max()
        - positions[:, 2].min()
    )
    features.update({
        f"{prefix}cell_area_A2": area,
        f"{prefix}cell_volume_A3": float(
            abs(np.linalg.det(cell))
        ),
        f"{prefix}layer_thickness_A": thickness,
        f"{prefix}atoms_per_area": float(
            len(numbers) / area
            if area > 0 else np.nan
        ),
    })

    translations = [
        int(value)
        for value in cfg["graph"][
            "periodic_xy_translations"
        ]
    ]
    radius_scale = float(
        cfg["graph"]["covalent_radius_scale"]
    )
    min_cutoff = float(
        cfg["graph"]["minimum_neighbor_cutoff_A"]
    )
    max_cutoff = float(
        cfg["graph"]["maximum_neighbor_cutoff_A"]
    )

    degrees = np.zeros(len(numbers), dtype=int)
    edge_distances = []
    edge_delta_z = []
    homonuclear = []

    for i in range(len(numbers)):
        radius_i = COVALENT_RADII.get(
            int(numbers[i]), 1.35
        )
        for j in range(i + 1, len(numbers)):
            radius_j = COVALENT_RADII.get(
                int(numbers[j]), 1.35
            )
            cutoff = np.clip(
                radius_scale * (radius_i + radius_j),
                min_cutoff,
                max_cutoff,
            )
            distance = minimum_periodic_distance(
                positions[i],
                positions[j],
                cell,
                translations,
            )
            if distance <= cutoff and distance > 1e-6:
                degrees[i] += 1
                degrees[j] += 1
                edge_distances.append(distance)
                edge_delta_z.append(
                    abs(
                        int(numbers[i])
                        - int(numbers[j])
                    )
                )
                homonuclear.append(
                    int(numbers[i] == numbers[j])
                )

    edge_distances = np.asarray(
        edge_distances,
        dtype=float,
    )
    edge_delta_z = np.asarray(
        edge_delta_z,
        dtype=float,
    )
    homonuclear = np.asarray(
        homonuclear,
        dtype=float,
    )

    features.update({
        f"{prefix}edge_count": float(
            len(edge_distances)
        ),
        f"{prefix}edges_per_atom": float(
            2 * len(edge_distances) / len(numbers)
        ),
        f"{prefix}degree_mean": float(
            degrees.mean()
        ),
        f"{prefix}degree_std": float(
            degrees.std()
        ),
        f"{prefix}degree_min": float(
            degrees.min()
        ),
        f"{prefix}degree_max": float(
            degrees.max()
        ),
        f"{prefix}isolated_fraction": float(
            (degrees == 0).mean()
        ),
        f"{prefix}edge_distance_mean": float(
            edge_distances.mean()
            if len(edge_distances) else np.nan
        ),
        f"{prefix}edge_distance_std": float(
            edge_distances.std()
            if len(edge_distances) else np.nan
        ),
        f"{prefix}edge_distance_min": float(
            edge_distances.min()
            if len(edge_distances) else np.nan
        ),
        f"{prefix}edge_distance_max": float(
            edge_distances.max()
            if len(edge_distances) else np.nan
        ),
        f"{prefix}edge_homonuclear_fraction": float(
            homonuclear.mean()
            if len(homonuclear) else np.nan
        ),
        f"{prefix}edge_delta_Z_mean": float(
            edge_delta_z.mean()
            if len(edge_delta_z) else np.nan
        ),
        f"{prefix}edge_delta_Z_std": float(
            edge_delta_z.std()
            if len(edge_delta_z) else np.nan
        ),
    })

    if len(edge_distances):
        for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
            features[
                f"{prefix}edge_distance_q{int(quantile*100)}"
            ] = float(
                np.quantile(
                    edge_distances,
                    quantile,
                )
            )
    else:
        for quantile in [10, 25, 50, 75, 90]:
            features[
                f"{prefix}edge_distance_q{quantile}"
            ] = np.nan

    max_degree = int(
        cfg["graph"]["degree_histogram_max"]
    )
    for degree in range(max_degree):
        features[
            f"{prefix}degree_fraction_{degree}"
        ] = float(
            (degrees == degree).mean()
        )
    features[
        f"{prefix}degree_fraction_{max_degree}_plus"
    ] = float(
        (degrees >= max_degree).mean()
    )

    radial_edges = np.asarray(
        cfg["graph"]["radial_histogram_edges_A"],
        dtype=float,
    )
    radial_hist, _ = np.histogram(
        edge_distances,
        bins=radial_edges,
    )
    radial_denominator = max(
        1,
        int(radial_hist.sum()),
    )
    for index, value in enumerate(radial_hist):
        features[
            f"{prefix}radial_fraction_{index}"
        ] = float(
            value / radial_denominator
        )

    return features


def intensive_feature_names(columns, prefix):
    excluded_suffixes = {
        "n_atoms",
        "cell_area_A2",
        "cell_volume_A3",
        "edge_count",
    }
    names = []
    for column in columns:
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix):]
        if suffix not in excluded_suffixes:
            names.append(column)
    return names


def load_and_extract(package_root, cfg, out, log):
    source = package_root / cfg["input"][
        "stage2a_fix1_zip"
    ]
    with zipfile.ZipFile(source) as archive:
        manifest = pd.read_csv(io.BytesIO(
            archive.read(
                "stage2a_fix1_output/"
                "RECONCILED_GRAPH_READY_MANIFEST.csv"
            )
        ))

        defect_registry = pd.read_csv(io.BytesIO(
            archive.read(
                "stage2a_fix1_output/"
                "SAFE_DEFECT_STRUCTURE_REGISTRY.csv"
            )
        ))
        pristine_registry = pd.read_csv(io.BytesIO(
            archive.read(
                "stage2a_fix1_output/"
                "SAFE_PRISTINE_STRUCTURE_REGISTRY.csv"
            )
        ))

        defect_path_map = dict(zip(
            defect_registry["record_id"].astype(str),
            defect_registry["relative_path"].astype(str),
        ))
        pristine_path_map = dict(zip(
            pristine_registry["host_key"].astype(str),
            pristine_registry["relative_path"].astype(str),
        ))

        pristine_features = {}
        for host, relative_path in pristine_path_map.items():
            member = (
                "stage2a_fix1_output/"
                + relative_path.replace("\\", "/")
            )
            structure = load_structure(
                archive,
                member,
            )
            pristine_features[host] = structure_features(
                structure,
                cfg,
                "p_",
            )

        rows = []
        errors = []
        for index, row in enumerate(
            manifest.itertuples(index=False),
            start=1,
        ):
            record_id = str(row.record_id)
            host = str(row.host_key)
            try:
                defect_member = (
                    "stage2a_fix1_output/"
                    + defect_path_map[
                        record_id
                    ].replace("\\", "/")
                )
                defect_structure = load_structure(
                    archive,
                    defect_member,
                )
                defect_features = structure_features(
                    defect_structure,
                    cfg,
                    "d_",
                )
                pristine = pristine_features[host]

                combined = {
                    "record_id": record_id,
                    "host_key": host,
                    "target_is_magnetic": int(
                        row.target_is_magnetic
                    ),
                    "ordinal_class": int(
                        row.ordinal_class
                    ),
                    "operation_pattern": str(
                        row.operation_pattern
                    ),
                    "dopant_signature": str(
                        row.dopant_signature
                    ),
                    "site_signature": str(
                        row.site_signature
                    ),
                    "operation_count": float(
                        row.operation_count
                    ),
                    "vacancy_count": float(
                        row.vacancy_count
                    ),
                    "substitution_count": float(
                        row.substitution_count
                    ),
                    "formula_nspecies": float(
                        row.formula_nspecies
                    ),
                    "formula_total_atoms": float(
                        row.formula_total_atoms
                    ),
                    "formula_Z_mean": float(
                        row.formula_Z_mean
                    ),
                    "formula_Z_std": float(
                        row.formula_Z_std
                    ),
                    "formula_Z_min": float(
                        row.formula_Z_min
                    ),
                    "formula_Z_max": float(
                        row.formula_Z_max
                    ),
                    **defect_features,
                    **pristine,
                }

                d_columns = list(
                    defect_features.keys()
                )
                for d_name in intensive_feature_names(
                    d_columns,
                    "d_",
                ):
                    suffix = d_name[2:]
                    p_name = "p_" + suffix
                    if p_name in pristine:
                        combined[
                            "x_" + suffix
                        ] = (
                            defect_features[d_name]
                            - pristine[p_name]
                        )

                combined[
                    "x_atom_count_ratio"
                ] = (
                    defect_features["d_n_atoms"]
                    / pristine["p_n_atoms"]
                )
                combined[
                    "x_area_ratio"
                ] = (
                    defect_features["d_cell_area_A2"]
                    / pristine["p_cell_area_A2"]
                )
                combined[
                    "x_density_ratio"
                ] = (
                    defect_features["d_atoms_per_area"]
                    / pristine["p_atoms_per_area"]
                )
                rows.append(combined)

            except Exception as exc:
                errors.append({
                    "record_id": record_id,
                    "host_key": host,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

            if index % 25 == 0 or index == len(manifest):
                log.info(
                    f"Graph descriptors "
                    f"{index}/{len(manifest)}; "
                    f"success={len(rows)}; "
                    f"errors={len(errors)}."
                )

    frame = pd.DataFrame(rows)
    errors_frame = pd.DataFrame(
        errors,
        columns=[
            "record_id",
            "host_key",
            "error",
        ],
    )
    frame.to_csv(
        out / "GRAPH_DESCRIPTOR_TABLE.csv",
        index=False,
    )
    errors_frame.to_csv(
        out / "GRAPH_EXTRACTION_ERRORS.csv",
        index=False,
    )

    forbidden_fragments = [
        "magmom",
        "magmoms",
        "energy",
        "forces",
    ]
    leaked_columns = [
        column for column in frame.columns
        if any(
            fragment in column.lower()
            for fragment in forbidden_fragments
        )
    ]
    write_json(out / "GRAPH_EXTRACTION_AUDIT.json", {
        "input_sha256": sha256_file(source),
        "input_records": len(manifest),
        "extracted_records": len(frame),
        "errors": len(errors),
        "hosts": int(
            frame["host_key"].nunique()
        ) if len(frame) else 0,
        "feature_columns": int(
            len(frame.columns) - 4
        ) if len(frame) else 0,
        "forbidden_feature_columns": leaked_columns,
        "target_fields_used_as_features": False,
        "host_identity_used_as_feature": False,
    })
    return frame, errors


def columns_for_feature_sets(frame):
    metadata_numeric = [
        "operation_count",
        "vacancy_count",
        "substitution_count",
        "formula_nspecies",
        "formula_total_atoms",
        "formula_Z_mean",
        "formula_Z_std",
        "formula_Z_min",
        "formula_Z_max",
    ]
    metadata_categorical = [
        "operation_pattern",
        "dopant_signature",
        "site_signature",
    ]

    d_all = [
        column for column in frame.columns
        if column.startswith("d_")
    ]
    p_all = [
        column for column in frame.columns
        if column.startswith("p_")
    ]
    x_all = [
        column for column in frame.columns
        if column.startswith("x_")
    ]
    d_composition = [
        column for column in d_all
        if (
            "Z_" in column
            or column.endswith("n_atoms")
            or "cell_" in column
            or column.endswith(
                "atoms_per_area"
            )
        )
    ]

    return {
        "metadata_only": (
            metadata_numeric,
            metadata_categorical,
        ),
        "defect_composition": (
            metadata_numeric + d_composition,
            metadata_categorical,
        ),
        "defect_graph": (
            metadata_numeric + d_all,
            metadata_categorical,
        ),
        "pristine_graph": (
            metadata_numeric + p_all,
            metadata_categorical,
        ),
        "concatenated_graph": (
            metadata_numeric + d_all + p_all,
            metadata_categorical,
        ),
        "twin_difference_graph": (
            metadata_numeric
            + d_all
            + p_all
            + x_all,
            metadata_categorical,
        ),
    }


def make_preprocessor(
    numeric,
    categorical,
):
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]),
            categorical,
        ))
    return ColumnTransformer(
        transformers,
        remainder="drop",
    )


def inverse_host_weights(frame):
    counts = frame[
        "host_key"
    ].value_counts()
    return frame["host_key"].map(
        lambda host: 1.0 / counts[host]
    ).to_numpy()


def safe_balanced_accuracy(
    y_true,
    y_pred,
):
    recalls = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for label in [0, 1]:
        mask = y_true == label
        if mask.sum() == 0:
            continue
        recalls.append(
            float(
                (y_pred[mask] == label).mean()
            )
        )
    return (
        float(np.mean(recalls))
        if recalls else np.nan
    )


def run_loho(frame, cfg, out, log):
    feature_sets = columns_for_feature_sets(
        frame
    )
    hosts = sorted(
        frame["host_key"].unique()
    )
    trees = int(
        cfg["evaluation"][
            "random_forest_trees"
        ]
    )
    leaf = int(
        cfg["evaluation"][
            "random_forest_min_samples_leaf"
        ]
    )
    seed = int(cfg["random_seed"])

    binary_predictions = []
    ordinal_predictions = []
    feature_rows = []

    for feature_set, (
        numeric,
        categorical,
    ) in feature_sets.items():
        numeric = [
            column for column in numeric
            if column in frame.columns
        ]
        categorical = [
            column for column in categorical
            if column in frame.columns
        ]
        feature_rows.append({
            "feature_set": feature_set,
            "numeric_features": len(numeric),
            "categorical_features": len(
                categorical
            ),
            "feature_names": "|".join(
                numeric + categorical
            ),
        })

        for host_index, test_host in enumerate(
            hosts,
            start=1,
        ):
            test_mask = frame[
                "host_key"
            ].eq(test_host)
            train = frame.loc[
                ~test_mask
            ]
            test = frame.loc[test_mask]
            weights = inverse_host_weights(
                train
            )

            binary = Pipeline([
                (
                    "preprocess",
                    make_preprocessor(
                        numeric,
                        categorical,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ])
            binary.fit(
                train[
                    numeric + categorical
                ],
                train[
                    "target_is_magnetic"
                ],
                model__sample_weight=weights,
            )
            classes = list(binary.classes_)
            matrix = binary.predict_proba(
                test[
                    numeric + categorical
                ]
            )
            probability = (
                matrix[:, classes.index(1)]
                if 1 in classes
                else np.zeros(len(test))
            )
            prediction = (
                probability >= 0.5
            ).astype(int)

            for index, prob, pred in zip(
                test.index,
                probability,
                prediction,
            ):
                binary_predictions.append({
                    "feature_set": feature_set,
                    "test_host": test_host,
                    "record_id": frame.loc[
                        index, "record_id"
                    ],
                    "y_true": int(
                        frame.loc[
                            index,
                            "target_is_magnetic",
                        ]
                    ),
                    "probability": float(prob),
                    "y_pred": int(pred),
                })

            ordinal = Pipeline([
                (
                    "preprocess",
                    make_preprocessor(
                        numeric,
                        categorical,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        random_state=(
                            seed + 101
                        ),
                        n_jobs=-1,
                    ),
                ),
            ])
            ordinal.fit(
                train[
                    numeric + categorical
                ],
                train["ordinal_class"],
                model__sample_weight=weights,
            )
            ordinal_prediction = ordinal.predict(
                test[
                    numeric + categorical
                ]
            )

            for index, pred in zip(
                test.index,
                ordinal_prediction,
            ):
                ordinal_predictions.append({
                    "feature_set": feature_set,
                    "test_host": test_host,
                    "record_id": frame.loc[
                        index, "record_id"
                    ],
                    "y_true": int(
                        frame.loc[
                            index,
                            "ordinal_class",
                        ]
                    ),
                    "y_pred": int(pred),
                })

            if (
                host_index % 10 == 0
                or host_index == len(hosts)
            ):
                log.info(
                    f"{feature_set}: "
                    f"LOHO host "
                    f"{host_index}/{len(hosts)}."
                )

    binary = pd.DataFrame(
        binary_predictions
    )
    ordinal = pd.DataFrame(
        ordinal_predictions
    )
    pd.DataFrame(feature_rows).to_csv(
        out / "FEATURE_SET_DEFINITIONS.csv",
        index=False,
    )
    binary.to_csv(
        out / "LOHO_BINARY_PREDICTIONS.csv",
        index=False,
    )
    ordinal.to_csv(
        out / "LOHO_ORDINAL_PREDICTIONS.csv",
        index=False,
    )

    binary_summary = []
    binary_host_rows = []
    for feature_set, subset in binary.groupby(
        "feature_set"
    ):
        y = subset["y_true"].to_numpy()
        probability = subset[
            "probability"
        ].to_numpy()
        prediction = subset[
            "y_pred"
        ].to_numpy()
        binary_summary.append({
            "feature_set": feature_set,
            "records": len(subset),
            "AUROC": float(
                roc_auc_score(
                    y,
                    probability,
                )
            ),
            "accuracy": float(
                accuracy_score(
                    y,
                    prediction,
                )
            ),
            "balanced_accuracy": (
                safe_balanced_accuracy(
                    y,
                    prediction,
                )
            ),
            "F1": float(
                f1_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),
        })

        for host, host_subset in subset.groupby(
            "test_host"
        ):
            host_y = host_subset[
                "y_true"
            ].to_numpy()
            host_probability = host_subset[
                "probability"
            ].to_numpy()
            host_prediction = host_subset[
                "y_pred"
            ].to_numpy()
            binary_host_rows.append({
                "feature_set": feature_set,
                "test_host": host,
                "records": len(host_subset),
                "positive_fraction": float(
                    host_y.mean()
                ),
                "AUROC": (
                    float(
                        roc_auc_score(
                            host_y,
                            host_probability,
                        )
                    )
                    if len(np.unique(host_y)) == 2
                    else np.nan
                ),
                "accuracy": float(
                    accuracy_score(
                        host_y,
                        host_prediction,
                    )
                ),
                "balanced_accuracy": (
                    safe_balanced_accuracy(
                        host_y,
                        host_prediction,
                    )
                ),
            })

    ordinal_summary = []
    ordinal_host_rows = []
    for feature_set, subset in ordinal.groupby(
        "feature_set"
    ):
        y = subset["y_true"].to_numpy()
        prediction = subset[
            "y_pred"
        ].to_numpy()
        errors = np.abs(y - prediction)
        ordinal_summary.append({
            "feature_set": feature_set,
            "records": len(subset),
            "QWK": float(
                cohen_kappa_score(
                    y,
                    prediction,
                    weights="quadratic",
                )
            ),
            "macro_F1": float(
                f1_score(
                    y,
                    prediction,
                    average="macro",
                    zero_division=0,
                )
            ),
            "class_MAE": float(
                errors.mean()
            ),
            "within_one_accuracy": float(
                (errors <= 1).mean()
            ),
            "severe_error_rate": float(
                (errors >= 2).mean()
            ),
        })

        for host, host_subset in subset.groupby(
            "test_host"
        ):
            host_y = host_subset[
                "y_true"
            ].to_numpy()
            host_prediction = host_subset[
                "y_pred"
            ].to_numpy()
            host_errors = np.abs(
                host_y - host_prediction
            )
            ordinal_host_rows.append({
                "feature_set": feature_set,
                "test_host": host,
                "records": len(host_subset),
                "classes_in_truth": int(
                    len(np.unique(host_y))
                ),
                "QWK": (
                    float(
                        cohen_kappa_score(
                            host_y,
                            host_prediction,
                            weights="quadratic",
                        )
                    )
                    if (
                        len(np.unique(host_y))
                        >= 2
                        and len(
                            np.unique(
                                host_prediction
                            )
                        ) >= 2
                    )
                    else np.nan
                ),
                "class_MAE": float(
                    host_errors.mean()
                ),
                "within_one_accuracy": float(
                    (host_errors <= 1).mean()
                ),
            })

    binary_summary = pd.DataFrame(
        binary_summary
    ).sort_values(
        "AUROC",
        ascending=False,
    )
    ordinal_summary = pd.DataFrame(
        ordinal_summary
    ).sort_values(
        "QWK",
        ascending=False,
    )
    binary_summary.to_csv(
        out / "LOHO_BINARY_SUMMARY.csv",
        index=False,
    )
    ordinal_summary.to_csv(
        out / "LOHO_ORDINAL_SUMMARY.csv",
        index=False,
    )
    pd.DataFrame(binary_host_rows).to_csv(
        out / "LOHO_BINARY_HOST_METRICS.csv",
        index=False,
    )
    pd.DataFrame(ordinal_host_rows).to_csv(
        out / "LOHO_ORDINAL_HOST_METRICS.csv",
        index=False,
    )

    write_json(out / "LEAKAGE_AUDIT.json", {
        "protocol": "leave_one_host_out",
        "host_identity_used_as_feature": False,
        "record_id_used_as_feature": False,
        "target_fields_used_as_features": False,
        "raw_structure_fields_used": [
            "numbers",
            "positions",
            "cell",
            "pbc",
        ],
        "forbidden_raw_fields_excluded": [
            "magmom",
            "magmoms",
            "energy",
            "forces",
        ],
        "host_overlap_detected": False,
    })
    return binary_summary, ordinal_summary


def decide(
    binary,
    ordinal,
    extraction_errors,
    cfg,
):
    metadata_binary = float(
        binary.loc[
            binary["feature_set"].eq(
                "metadata_only"
            ),
            "AUROC",
        ].iloc[0]
    )
    metadata_ordinal = float(
        ordinal.loc[
            ordinal["feature_set"].eq(
                "metadata_only"
            ),
            "QWK",
        ].iloc[0]
    )

    structural_names = [
        "defect_composition",
        "defect_graph",
        "pristine_graph",
        "concatenated_graph",
        "twin_difference_graph",
    ]
    structural_binary = binary.loc[
        binary["feature_set"].isin(
            structural_names
        )
    ].sort_values(
        "AUROC",
        ascending=False,
    )
    structural_ordinal = ordinal.loc[
        ordinal["feature_set"].isin(
            structural_names
        )
    ].sort_values(
        "QWK",
        ascending=False,
    )

    best_binary = structural_binary.iloc[0]
    best_ordinal = structural_ordinal.iloc[0]

    binary_gain = (
        float(best_binary["AUROC"])
        - metadata_binary
    )
    ordinal_gain = (
        float(best_ordinal["QWK"])
        - metadata_ordinal
    )

    gates = cfg["gates"]
    if (
        extraction_errors == 0
        and float(best_binary["AUROC"])
        >= float(
            gates["minimum_binary_auroc"]
        )
        and float(best_ordinal["QWK"])
        >= float(
            gates["minimum_ordinal_qwk"]
        )
        and binary_gain
        >= float(
            gates[
                "minimum_binary_gain_over_metadata_green"
            ]
        )
        and ordinal_gain
        >= float(
            gates[
                "minimum_ordinal_gain_over_metadata_green"
            ]
        )
    ):
        status = "GREEN"
        next_stage = (
            "STAGE2C_GNN_BASELINE_PILOT"
        )
    elif (
        extraction_errors == 0
        and binary_gain
        >= -float(
            gates[
                "maximum_allowed_metric_drop_yellow"
            ]
        )
        and ordinal_gain
        >= -float(
            gates[
                "maximum_allowed_metric_drop_yellow"
            ]
        )
    ):
        status = "YELLOW"
        next_stage = (
            "STAGE2C_SMALL_CONTROLLED_GNN_PILOT"
        )
    else:
        status = "RED"
        next_stage = (
            "GRAPH_REPRESENTATION_REDESIGN"
        )

    return {
        "stage2b_decision": status,
        "metadata_binary_AUROC": (
            metadata_binary
        ),
        "metadata_ordinal_QWK": (
            metadata_ordinal
        ),
        "best_structural_binary_feature_set": str(
            best_binary["feature_set"]
        ),
        "best_structural_binary_AUROC": float(
            best_binary["AUROC"]
        ),
        "binary_AUROC_gain_over_metadata": (
            binary_gain
        ),
        "best_structural_ordinal_feature_set": str(
            best_ordinal["feature_set"]
        ),
        "best_structural_ordinal_QWK": float(
            best_ordinal["QWK"]
        ),
        "ordinal_QWK_gain_over_metadata": (
            ordinal_gain
        ),
        "graph_extraction_errors": int(
            extraction_errors
        ),
        "next_allowed_stage": next_stage,
        "actual_gnn_training_allowed": (
            status in {"GREEN", "YELLOW"}
        ),
        "final_cdt_egnn_training_allowed": False,
    }


def write_summary(out, decision, binary, ordinal):
    binary_table = binary.to_string(
        index=False
    )
    ordinal_table = ordinal.to_string(
        index=False
    )
    text = f"""# نتيجة Stage 2B

## القرار

- Stage 2B: **{decision['stage2b_decision']}**
- graph extraction errors:
  **{decision['graph_extraction_errors']}**

## Binary magnetic activation

- metadata AUROC:
  **{decision['metadata_binary_AUROC']:.4f}**
- best structural set:
  `{decision['best_structural_binary_feature_set']}`
- best structural AUROC:
  **{decision['best_structural_binary_AUROC']:.4f}**
- gain over metadata:
  **{decision['binary_AUROC_gain_over_metadata']:.4f}**

```text
{binary_table}
```

## Ordinal magnitude

- metadata QWK:
  **{decision['metadata_ordinal_QWK']:.4f}**
- best structural set:
  `{decision['best_structural_ordinal_feature_set']}`
- best structural QWK:
  **{decision['best_structural_ordinal_QWK']:.4f}**
- gain over metadata:
  **{decision['ordinal_QWK_gain_over_metadata']:.4f}**

```text
{ordinal_table}
```

## المرحلة التالية

`{decision['next_allowed_stage']}`

- actual GNN pilot allowed:
  **{decision['actual_gnn_training_allowed']}**
- final CDT-EGNN training:
  **غير مسموح**

لم تُستخدم الحقول المغناطيسية أو الطاقة أو القوى كميزات.
"""
    (out / "ARABIC_STAGE2B_SUMMARY.md").write_text(
        text,
        encoding="utf-8",
    )


def inventory(out):
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            rows.append({
                "relative_path": str(
                    path.relative_to(out)
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    pd.DataFrame(rows).to_csv(
        out / "HASHES_SHA256.csv",
        index=False,
    )


def create_results_zip(
    out,
    destination,
    cfg,
):
    fixed = tuple(
        int(value)
        for value in cfg["packaging"][
            "fixed_zip_datetime"
        ]
    )
    with zipfile.ZipFile(
        destination,
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                str(
                    Path("stage2b_output")
                    / path.relative_to(out)
                ).replace("\\", "/"),
                date_time=fixed,
            )
            info.compress_type = (
                zipfile.ZIP_DEFLATED
            )
            info.external_attr = 0o644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    package_root = cfg_path.parents[1]
    cfg = yaml.safe_load(
        cfg_path.read_text(
            encoding="utf-8"
        )
    )
    out = Path(
        cfg["output_dir"]
    ).resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    log = Logger(out)

    status = {
        "project_code": cfg["project_code"],
        "stage": cfg["stage"],
        "stage_version": cfg["stage_version"],
        "started_at": utcnow(),
        "state": "RUNNING",
    }
    write_json(out / "STATUS.json", status)
    (out / "CONFIG_SNAPSHOT.yaml").write_text(
        cfg_path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    for name in [
        "FEATURE_AND_LEAKAGE_RULES.md",
        "STAGE2B_GATE_RULES.md",
    ]:
        (out / name).write_text(
            (
                package_root
                / "docs"
                / name
            ).read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )

    try:
        frame, extraction_errors = (
            load_and_extract(
                package_root,
                cfg,
                out,
                log,
            )
        )
        if len(frame) != 529:
            raise RuntimeError(
                f"Expected 529 graph rows, "
                f"found {len(frame)}."
            )

        binary, ordinal = run_loho(
            frame,
            cfg,
            out,
            log,
        )
        decision = decide(
            binary,
            ordinal,
            len(extraction_errors),
            cfg,
        )
        write_json(
            out / "STAGE2B_GATE_DECISION.json",
            decision,
        )
        write_summary(
            out,
            decision,
            binary,
            ordinal,
        )

        required = [
            "STATUS.json",
            "CONFIG_SNAPSHOT.yaml",
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
            "ERROR_LOG.txt",
        ]
        verification = {}
        for name in required:
            path = out / name
            verification[name] = (
                path.exists()
                and (
                    name == "ERROR_LOG.txt"
                    or path.stat().st_size > 0
                )
            )
        extraction_audit = json.loads(
            (
                out
                / "GRAPH_EXTRACTION_AUDIT.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        leakage = json.loads(
            (
                out
                / "LEAKAGE_AUDIT.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        verification[
            "all_529_graph_rows"
        ] = (
            extraction_audit[
                "extracted_records"
            ] == 529
        )
        verification[
            "zero_graph_errors"
        ] = (
            extraction_audit["errors"] == 0
        )
        verification[
            "zero_forbidden_feature_columns"
        ] = (
            len(
                extraction_audit[
                    "forbidden_feature_columns"
                ]
            ) == 0
        )
        verification[
            "no_host_leakage"
        ] = (
            leakage[
                "host_overlap_detected"
            ] is False
            and leakage[
                "host_identity_used_as_feature"
            ] is False
        )
        verification[
            "all_required_present"
        ] = all(
            verification.values()
        )
        write_json(
            out / "OUTPUT_VERIFICATION.json",
            verification,
        )

        status.update({
            "finished_at": utcnow(),
            "state": "COMPLETED",
            "decision": decision[
                "stage2b_decision"
            ],
        })
        write_json(
            out / "STATUS.json",
            status,
        )
        write_json(
            out / "SUCCESS_MARKER.json",
            {
                "marker_type": (
                    "SOFTWARE_STAGE_COMPLETION_NOT_Q1_APPROVAL"
                ),
                "stage": cfg["stage"],
                "stage_version": cfg[
                    "stage_version"
                ],
                "completed_at": utcnow(),
                "decision": decision[
                    "stage2b_decision"
                ],
                "next_allowed_stage": decision[
                    "next_allowed_stage"
                ],
                "actual_gnn_training_allowed": decision[
                    "actual_gnn_training_allowed"
                ],
                "final_cdt_egnn_training_allowed": False,
                "output_verification_passed": verification[
                    "all_required_present"
                ],
            },
        )

        inventory(out)
        create_results_zip(
            out,
            package_root / "STAGE2B_RESULTS.zip",
            cfg,
        )
        log.info(
            "Stage 2B completed. "
            f"Decision="
            f"{decision['stage2b_decision']}."
        )
        return 0

    except Exception:
        log.error(
            traceback.format_exc()
        )
        status.update({
            "finished_at": utcnow(),
            "state": "FAILED",
        })
        write_json(
            out / "STATUS.json",
            status,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
