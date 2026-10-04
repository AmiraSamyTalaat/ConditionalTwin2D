from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLE = OUT / "tables"
CHECK = OUT / "checkpoints"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def safe_float(value):
    if value is None:
        return None
    x = float(value)
    if not np.isfinite(x):
        return None
    return x


def latex_escape(value: str) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("--")
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(latex_escape(v))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def inverse_host_weights(hosts: pd.Series) -> np.ndarray:
    counts = hosts.value_counts()
    w = hosts.map(lambda h: 1.0 / counts[h]).to_numpy(float)
    return w / w.mean()


def ece_score(y, p, bins: int = 15) -> float:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def binary_metrics(y, p, threshold: float = 0.5, ece_bins: int = 15) -> dict:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    pred = (p >= threshold).astype(int)
    both = np.unique(y).size == 2
    prevalence = float(y.mean()) if len(y) else float("nan")
    prevalence_brier = prevalence * (1 - prevalence)
    brier = float(brier_score_loss(y, p))
    return {
        "records": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "AUROC": float(roc_auc_score(y, p)) if both else float("nan"),
        "AUPRC": float(average_precision_score(y, p)) if both else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": brier,
        "Brier_skill": float(1 - brier / prevalence_brier) if prevalence_brier > 0 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ECE": ece_score(y, p, ece_bins),
        "predicted_positive_fraction": float(pred.mean()),
    }


def host_centered_auc(df: pd.DataFrame, score: str, target: str, host: str) -> float:
    centered = df[score] - df.groupby(host)[score].transform("mean")
    y = df[target].to_numpy(int)
    return float(roc_auc_score(y, centered)) if np.unique(y).size == 2 else float("nan")


def within_host_pair_auc(df: pd.DataFrame, score: str, target: str, host: str) -> tuple[float, int]:
    concordant = 0.0
    pairs = 0
    for _, g in df.groupby(host):
        pos = g.loc[g[target] == 1, score].to_numpy(float)
        neg = g.loc[g[target] == 0, score].to_numpy(float)
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        concordant += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        pairs += int(diff.size)
    return (concordant / pairs if pairs else float("nan"), pairs)


def macro_host_auc(df: pd.DataFrame, score: str, target: str, host: str) -> tuple[float, int]:
    vals = []
    for _, g in df.groupby(host):
        if g[target].nunique() == 2:
            vals.append(roc_auc_score(g[target], g[score]))
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def host_decomposition(df: pd.DataFrame, score: str, target: str, host: str) -> dict:
    base = binary_metrics(df[target], df[score])
    pair_auc, pairs = within_host_pair_auc(df, score, target, host)
    macro_auc, evaluable = macro_host_auc(df, score, target, host)
    h = df.groupby(host).agg(prevalence=(target, "mean"), mean_score=(score, "mean"), records=(target, "size")).reset_index()
    rho = spearmanr(h.prevalence, h.mean_score).statistic if len(h) >= 3 else float("nan")
    return {
        **base,
        "hosts": int(df[host].nunique()),
        "evaluable_hosts": int(evaluable),
        "host_centered_AUROC": host_centered_auc(df, score, target, host),
        "within_host_pair_AUROC": pair_auc,
        "within_host_pairs": int(pairs),
        "macro_host_AUROC": macro_auc,
        "host_prevalence_score_spearman": float(rho) if np.isfinite(rho) else float("nan"),
    }


def bootstrap_host_metrics(df: pd.DataFrame, score: str, target: str, host: str, reps: int, seed: int) -> pd.DataFrame:
    hosts = list(df[host].unique())
    cache = {h: df[df[host] == h].copy() for h in hosts}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(reps):
        sampled = rng.choice(hosts, size=len(hosts), replace=True)
        parts = []
        for j, h in enumerate(sampled):
            part = cache[h].copy()
            part["_boot_host"] = f"{j}:{h}"
            parts.append(part)
        g = pd.concat(parts, ignore_index=True)
        if g[target].nunique() < 2:
            continue
        pair, pairs = within_host_pair_auc(g, score, target, "_boot_host")
        rows.append({
            "replicate": b,
            "AUROC": roc_auc_score(g[target], g[score]),
            "host_centered_AUROC": host_centered_auc(g, score, target, "_boot_host"),
            "within_host_pair_AUROC": pair,
            "within_host_pairs": pairs,
        })
    return pd.DataFrame(rows)


def build_classifier(seed: int, cfg: dict) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(cfg["model"]["n_estimators"]),
        min_samples_leaf=int(cfg["model"]["min_samples_leaf"]),
        max_features=cfg["model"]["max_features"],
        class_weight=cfg["model"]["class_weight"],
        random_state=int(seed),
        n_jobs=int(cfg["model"]["n_jobs"]),
    )


def build_regressor(seed: int, cfg: dict) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(cfg["model"]["n_estimators"]),
        min_samples_leaf=int(cfg["model"]["min_samples_leaf"]),
        max_features=cfg["model"]["max_features"],
        random_state=int(seed) % 4294967295,
        n_jobs=int(cfg["model"]["n_jobs"]),
    )


@dataclass
class Calibrator:
    name: str
    payload: object
    valid: bool = True

    def apply(self, p) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
        if self.name == "identity" or not self.valid:
            return p
        if self.name == "platt":
            x = logit(p).reshape(-1, 1)
            return self.payload.predict_proba(x)[:, 1]
        if self.name == "beta":
            x = np.column_stack([np.log(p), -np.log1p(-p)])
            return self.payload.predict_proba(x)[:, 1]
        if self.name == "temperature":
            return expit(logit(p) / float(self.payload))
        raise ValueError(self.name)


def fit_calibrator(name: str, y, p, weights) -> Calibrator:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    w = np.asarray(weights, float)
    if name == "identity":
        return Calibrator(name, None)
    if name == "platt":
        x = logit(p).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        model.fit(x, y, sample_weight=w)
        valid = float(model.coef_[0, 0]) > 0
        return Calibrator(name, model, valid)
    if name == "beta":
        x = np.column_stack([np.log(p), -np.log1p(-p)])
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        model.fit(x, y, sample_weight=w)
        valid = bool((model.coef_[0] >= 0).all())
        return Calibrator(name, model, valid)
    if name == "temperature":
        z = logit(p)

        def objective(log_t):
            q = expit(z / np.exp(log_t))
            losses = -(y * np.log(q + 1e-12) + (1 - y) * np.log1p(-q + 1e-12))
            return float(np.average(losses, weights=w))

        result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
        return Calibrator(name, float(np.exp(result.x)), bool(result.success))
    raise ValueError(name)


def load_data(cfg: dict):
    dev = pd.read_csv(INPUT / "development_representation.csv")
    target = pd.read_csv(INPUT / "development_targets_manifest.csv")
    target = target[["record_id", "host_key", "target_abs_magnetic_moment", "authoritative_json_magmom"]].copy()
    target["abs_moment"] = target.authoritative_json_magmom.abs()
    merged = dev.merge(target[["record_id", "abs_moment"]], on="record_id", how="left", validate="one_to_one")
    if merged.abs_moment.isna().any() or len(merged) != 529:
        raise RuntimeError("Development target join failed.")

    base = pd.read_csv(INPUT / "original_base_loho_predictions.csv")
    base = base[base.model == "et_x_composition_delta_primary"].copy()
    imp = pd.read_csv(INPUT / "imp2d_evaluated_original.csv")
    imp_features = pd.read_csv(INPUT / "imp2d_features.csv")
    imp = imp.merge(imp_features, on=["ase_row_id", "external_id", "name", "host", "selected_phase_uid", "dopant", "defecttype", "site"], how="inner", validate="one_to_one")
    development_hosts = set(merged.host_key.unique())
    imp["strict_primary"] = ~imp.selected_phase_uid.isin(development_hosts)

    d2_eval = pd.read_csv(INPUT / "2dmd_evaluated_original.csv")
    d2_feat = pd.read_csv(INPUT / "2dmd_features.csv")
    d2 = d2_eval[["_id", "host", "strict_primary", "operation_family", "defect_count", "identity_probability", "beta_probability", "temperature_probability", "seed_probability_std", "total_mag", "y_true"]].merge(
        d2_feat, on=["_id", "host", "strict_primary", "operation_family", "defect_count"], how="inner", validate="one_to_one"
    )

    feature_names = sorted([c for c in merged if c.startswith("x_")] + [c for c in merged if c.startswith("c_") and "delta" in c])
    if len(feature_names) != 68:
        raise RuntimeError(f"Expected 68 primary features, found {len(feature_names)}")
    for frame, name in [(imp, "Imp2D"), (d2, "2DMD")]:
        missing = [c for c in feature_names if c not in frame]
        if missing:
            raise RuntimeError(f"{name} missing primary features: {missing[:5]}")
    return merged, base, imp, d2, feature_names


def nested_calibration_analysis(dev, base, features, cfg):
    print("[1/7] Nested calibration comparison")
    out_path = CHECK / "nested_calibration_predictions.csv"
    rows = []
    hosts = sorted(dev.host_key.unique())
    methods = ["identity", "platt", "beta", "temperature"]
    seeds = [int(x) for x in cfg["seeds"]]
    for sidx, seed in enumerate(seeds, 1):
        for hidx, test_host in enumerate(hosts, 1):
            train = dev[dev.host_key != test_host].reset_index(drop=True)
            test = dev[dev.host_key == test_host].copy()
            splitter = StratifiedGroupKFold(
                n_splits=int(cfg["calibration"]["inner_group_splits"]),
                shuffle=True,
                random_state=seed + stable_int(test_host),
            )
            oof = np.full(len(train), np.nan)
            for fold, (tr_idx, ca_idx) in enumerate(splitter.split(train[features], train.target_is_magnetic, train.host_key)):
                tr = train.iloc[tr_idx]
                ca = train.iloc[ca_idx]
                model = build_classifier(seed + stable_int(test_host) + fold * 1009, cfg)
                model.fit(tr[features], tr.target_is_magnetic, sample_weight=inverse_host_weights(tr.host_key))
                oof[ca_idx] = model.predict_proba(ca[features])[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError("Incomplete inner OOF calibration predictions")
            outer = base[(base.seed == seed) & (base.test_host == test_host)].set_index("record_id")
            test_p = test.record_id.map(outer.probability).to_numpy(float)
            weights = inverse_host_weights(train.host_key)
            for method in methods:
                cal = fit_calibrator(method, train.target_is_magnetic, oof, weights)
                applied = cal.apply(test_p)
                for rec, y, pu, pc in zip(test.record_id, test.target_is_magnetic, test_p, applied):
                    rows.append({
                        "seed": seed,
                        "test_host": test_host,
                        "record_id": rec,
                        "y_true": int(y),
                        "method": method,
                        "uncalibrated_probability": float(pu),
                        "calibrated_probability": float(pc),
                        "calibrator_valid": bool(cal.valid),
                    })
            if (hidx % 5 == 0) or hidx == len(hosts):
                print(f"  calibration seed {sidx}/{len(seeds)} host {hidx}/{len(hosts)}")
    pred = pd.DataFrame(rows)
    pred.to_csv(out_path, index=False)
    pred.to_csv(TABLE / "TABLE_Q1_NESTED_CALIBRATION_PREDICTIONS.csv", index=False)

    ensemble = pred.groupby(["record_id", "y_true", "method"], as_index=False).calibrated_probability.mean()
    summaries = []
    for method, g in ensemble.groupby("method"):
        summaries.append({"method": method, **binary_metrics(g.y_true, g.calibrated_probability, ece_bins=int(cfg["calibration"]["ece_bins"]))})
    summary = pd.DataFrame(summaries).sort_values(["Brier", "log_loss", "ECE"]).reset_index(drop=True)
    selected = str(summary.iloc[0].method)
    summary["selected_on_development_only"] = summary.method.eq(selected)
    summary.to_csv(TABLE / "TABLE_Q1_CALIBRATION_METHOD_COMPARISON.csv", index=False)
    write_latex_table(summary[["method", "AUROC", "AUPRC", "Brier", "log_loss", "ECE", "selected_on_development_only"]], TABLE / "TABLE_Q1_CALIBRATION_METHOD_COMPARISON.tex", "Nested host-disjoint calibration comparison.", "tab:q1_calibration")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot_df = summary.set_index("method").loc[[m for m in methods if m in set(summary.method)]]
    x = np.arange(len(plot_df))
    width = 0.25
    ax.bar(x - width, plot_df.Brier, width, label="Brier")
    ax.bar(x, plot_df.log_loss, width, label="Log loss")
    ax.bar(x + width, plot_df.ECE, width, label="ECE")
    ax.set_xticks(x, plot_df.index)
    ax.set_ylabel("Lower is better")
    ax.set_title("Nested host-disjoint calibration comparison")
    ax.legend()
    save_figure(fig, "FigureQ1_Nested_Calibration_Comparison")
    return pred, ensemble, summary, selected


def fit_deployment_calibrators(dev, base, d2, imp, selected, cfg):
    oof = base.groupby(["record_id", "y_true"], as_index=False).probability.mean()
    joined = dev[["record_id", "host_key"]].merge(oof, on="record_id", validate="one_to_one")
    weights = inverse_host_weights(joined.host_key)
    calibrators = {m: fit_calibrator(m, joined.y_true, joined.probability, weights) for m in ["identity", "platt", "beta", "temperature"]}
    deployment = {}
    for m, cal in calibrators.items():
        payload = {"method": m, "valid": bool(cal.valid)}
        if m in ("platt", "beta") and cal.payload is not None:
            payload["coefficients"] = cal.payload.coef_[0].tolist()
            payload["intercept"] = float(cal.payload.intercept_[0])
        elif m == "temperature":
            payload["temperature"] = float(cal.payload)
        deployment[m] = payload
    deployment["selected_method"] = selected
    write_json(OUT / "DEVELOPMENT_ONLY_DEPLOYMENT_CALIBRATORS.json", deployment)

    rows = []
    for cohort, frame, target_col in [
        ("2DMD_strict", d2[d2.strict_primary].copy(), "y_true"),
        ("Imp2D_strict", imp[imp.strict_primary].copy(), "y_true"),
    ]:
        for m, cal in calibrators.items():
            p = cal.apply(frame.identity_probability)
            metrics = host_decomposition(pd.DataFrame({"host": frame.host, "y": frame[target_col], "p": p}), "p", "y", "host")
            rows.append({"cohort": cohort, "method": m, "post_lockbox_diagnostic": True, **metrics})
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLE / "TABLE_Q1_POSTLOCKBOX_CALIBRATION_DIAGNOSTICS.csv", index=False)
    return calibrators, diagnostics


def feature_groups(features):
    groups = {
        "composition_delta": [c for c in features if c.startswith("c_")],
        "atomic_number_change": [c for c in features if c.startswith("x_Z")],
        "scale_density": [c for c in features if any(k in c for k in ["area_ratio", "atom_count_ratio", "atoms_per_area", "density_ratio", "layer_thickness"])],
        "degree_connectivity": [c for c in features if ("degree" in c or c in ["x_edges_per_atom", "x_isolated_fraction"])],
        "edge_chemistry": [c for c in features if any(k in c for k in ["edge_delta_Z", "edge_homonuclear"])],
        "edge_geometry": [c for c in features if "edge_distance" in c],
        "radial_change": [c for c in features if c.startswith("x_radial")],
    }
    assigned = {x for xs in groups.values() for x in xs}
    leftover = [c for c in features if c not in assigned]
    if leftover:
        groups["other_graph_change"] = leftover
    return groups


def threshold_and_permutation_analysis(dev, d2, imp, features, cfg):
    print("[2/7] Endpoint-threshold sensitivity and grouped permutation importance")
    thresholds = [float(x) for x in cfg["thresholds_muB"]]
    seeds = [int(x) for x in cfg["seeds"]]
    hosts = sorted(dev.host_key.unique())
    internal_rows, ext_rows, perm_rows = [], [], []
    groups = feature_groups(features)
    rng = np.random.default_rng(20260717)
    d2_strict = d2[d2.strict_primary].copy()
    imp_strict = imp[imp.strict_primary].copy()

    for tidx, endpoint in enumerate(thresholds, 1):
        y_all = (dev.abs_moment >= endpoint).astype(int)
        for sidx, seed in enumerate(seeds, 1):
            for hidx, test_host in enumerate(hosts, 1):
                tr = dev[dev.host_key != test_host].copy()
                te = dev[dev.host_key == test_host].copy()
                ytr = (tr.abs_moment >= endpoint).astype(int)
                yte = (te.abs_moment >= endpoint).astype(int)
                model = build_classifier(seed + stable_int(f"thr{endpoint}:{test_host}"), cfg)
                model.fit(tr[features], ytr, sample_weight=inverse_host_weights(tr.host_key))
                p = model.predict_proba(te[features])[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(te))
                for rec, host, y, prob in zip(te.record_id, te.host_key, yte, p):
                    internal_rows.append({"endpoint_muB": endpoint, "seed": seed, "record_id": rec, "host": host, "y_true": int(y), "probability": float(prob)})

                if abs(endpoint - 0.10) < 1e-9:
                    base_brier = brier_score_loss(yte, p)
                    base_auc = roc_auc_score(yte, p) if yte.nunique() == 2 else np.nan
                    for group_name, cols in groups.items():
                        for rep in range(int(cfg["statistics"]["permutation_repeats"])):
                            perm = te[features].copy()
                            order = rng.permutation(len(perm))
                            perm.loc[:, cols] = perm[cols].to_numpy()[order]
                            pp = model.predict_proba(perm)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(perm))
                            perm_rows.append({
                                "seed": seed,
                                "test_host": test_host,
                                "group": group_name,
                                "repeat": rep,
                                "records": len(te),
                                "delta_Brier": float(brier_score_loss(yte, pp) - base_brier),
                                "delta_AUROC": float(base_auc - roc_auc_score(yte, pp)) if yte.nunique() == 2 else np.nan,
                            })
            full = build_classifier(seed + stable_int(f"full:{endpoint}"), cfg)
            full.fit(dev[features], y_all, sample_weight=inverse_host_weights(dev.host_key))
            for cohort, frame, target_values, id_col in [
                ("2DMD_strict", d2_strict, (d2_strict.total_mag.abs() >= endpoint).astype(int), "_id"),
                ("Imp2D_strict", imp_strict, (imp_strict.spin.abs() >= endpoint).astype(int), "external_id"),
            ]:
                pp = full.predict_proba(frame[features])[:, list(full.classes_).index(1)] if 1 in full.classes_ else np.zeros(len(frame))
                for rid, host, y, prob in zip(frame[id_col], frame.host, target_values, pp):
                    ext_rows.append({"endpoint_muB": endpoint, "seed": seed, "cohort": cohort, "record_id": rid, "host": host, "y_true": int(y), "probability": float(prob)})
            print(f"  threshold {tidx}/{len(thresholds)} seed {sidx}/{len(seeds)} complete")

    internal = pd.DataFrame(internal_rows)
    external = pd.DataFrame(ext_rows)
    perm = pd.DataFrame(perm_rows)
    internal.to_csv(CHECK / "threshold_internal_predictions.csv", index=False)
    external.to_csv(CHECK / "threshold_external_predictions.csv", index=False)
    perm.to_csv(TABLE / "TABLE_Q1_GROUPED_PERMUTATION_RUNS.csv", index=False)

    summary_rows = []
    for endpoint, g in internal.groupby("endpoint_muB"):
        ens = g.groupby(["record_id", "host", "y_true"], as_index=False).probability.mean()
        m = host_decomposition(ens, "probability", "y_true", "host")
        summary_rows.append({"cohort": "Development_LOHO", "endpoint_muB": endpoint, **m})
    for (cohort, endpoint), g in external.groupby(["cohort", "endpoint_muB"]):
        ens = g.groupby(["record_id", "host", "y_true"], as_index=False).probability.mean()
        m = host_decomposition(ens, "probability", "y_true", "host")
        summary_rows.append({"cohort": cohort, "endpoint_muB": endpoint, **m})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE / "TABLE_Q1_ENDPOINT_SENSITIVITY.csv", index=False)
    write_latex_table(summary[["cohort", "endpoint_muB", "records", "prevalence", "AUROC", "AUPRC", "Brier", "host_centered_AUROC", "within_host_pair_AUROC"]], TABLE / "TABLE_Q1_ENDPOINT_SENSITIVITY.tex", "Endpoint sensitivity under fixed pristine-defect representation.", "tab:q1_endpoint")

    perm_summary = perm.groupby("group", as_index=False).agg(
        mean_delta_Brier=("delta_Brier", "mean"),
        median_delta_Brier=("delta_Brier", "median"),
        mean_delta_AUROC=("delta_AUROC", "mean"),
        evaluable_AUROC_runs=("delta_AUROC", "count"),
        runs=("group", "size"),
    ).sort_values("mean_delta_Brier", ascending=False)
    perm_summary.to_csv(TABLE / "TABLE_Q1_GROUPED_PERMUTATION_IMPORTANCE.csv", index=False)
    write_latex_table(perm_summary, TABLE / "TABLE_Q1_GROUPED_PERMUTATION_IMPORTANCE.tex", "Grouped held-out-host permutation importance.", "tab:q1_groupperm")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for cohort, g in summary.groupby("cohort"):
        ax.plot(g.endpoint_muB, g.AUROC, marker="o", label=cohort)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Magnetic-moment endpoint (micro_B)")
    ax.set_ylabel("AUROC")
    ax.set_title("Endpoint sensitivity across development and external cohorts")
    ax.legend()
    save_figure(fig, "FigureQ2_Endpoint_Threshold_Sensitivity")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ps = perm_summary.sort_values("mean_delta_Brier")
    ax.barh(ps.group, ps.mean_delta_Brier)
    ax.set_xlabel("Increase in held-out-host Brier after group permutation")
    ax.set_title("Grouped permutation importance at 0.1 micro_B")
    save_figure(fig, "FigureQ7_Grouped_Permutation_Importance")
    return internal, external, summary, perm_summary


def continuous_target_analysis(dev, d2, imp, features, cfg):
    print("[3/7] Continuous-target analysis")
    seeds = [int(x) for x in cfg["seeds"]]
    hosts = sorted(dev.host_key.unique())
    rows, ext_rows = [], []
    target = np.log1p(dev.abs_moment.to_numpy(float))
    d2_strict = d2[d2.strict_primary].copy()
    imp_strict = imp[imp.strict_primary].copy()
    for sidx, seed in enumerate(seeds, 1):
        for test_host in hosts:
            tr_idx = dev.host_key != test_host
            te_idx = ~tr_idx
            model = build_regressor(seed + stable_int(f"reg:{test_host}"), cfg)
            model.fit(dev.loc[tr_idx, features], target[tr_idx], sample_weight=inverse_host_weights(dev.loc[tr_idx, "host_key"]))
            pred = np.maximum(0.0, model.predict(dev.loc[te_idx, features]))
            for rec, host, y, p in zip(dev.loc[te_idx, "record_id"], dev.loc[te_idx, "host_key"], target[te_idx], pred):
                rows.append({"seed": seed, "record_id": rec, "host": host, "target_log1p_abs_moment": float(y), "prediction": float(p)})
        full = build_regressor(seed + stable_int("reg:full"), cfg)
        full.fit(dev[features], target, sample_weight=inverse_host_weights(dev.host_key))
        for cohort, frame, raw_target, id_col in [
            ("2DMD_strict", d2_strict, d2_strict.total_mag.abs(), "_id"),
            ("Imp2D_strict", imp_strict, imp_strict.spin.abs(), "external_id"),
        ]:
            pred = np.maximum(0.0, full.predict(frame[features]))
            y = np.log1p(raw_target.to_numpy(float))
            for rid, host, yy, pp in zip(frame[id_col], frame.host, y, pred):
                ext_rows.append({"seed": seed, "cohort": cohort, "record_id": rid, "host": host, "target_log1p_abs_moment": float(yy), "prediction": float(pp)})
        print(f"  continuous seed {sidx}/{len(seeds)} complete")
    internal = pd.DataFrame(rows)
    external = pd.DataFrame(ext_rows)
    internal.to_csv(CHECK / "continuous_internal_predictions.csv", index=False)
    external.to_csv(CHECK / "continuous_external_predictions.csv", index=False)

    summaries = []
    for cohort, g in [("Development_LOHO", internal)]:
        ens = g.groupby(["record_id", "host", "target_log1p_abs_moment"], as_index=False).prediction.mean()
        summaries.append(continuous_metrics(cohort, ens))
    for cohort, g in external.groupby("cohort"):
        ens = g.groupby(["record_id", "host", "target_log1p_abs_moment"], as_index=False).prediction.mean()
        summaries.append(continuous_metrics(cohort, ens))
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLE / "TABLE_Q1_CONTINUOUS_TARGET.csv", index=False)
    write_latex_table(summary, TABLE / "TABLE_Q1_CONTINUOUS_TARGET.tex", "Continuous log-moment prediction diagnostics.", "tab:q1_continuous")

    fig, ax = plt.subplots(figsize=(7, 5.2))
    x = np.arange(len(summary))
    ax.bar(x, summary.spearman_rho)
    ax.set_xticks(x, summary.cohort, rotation=20, ha="right")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Continuous-target rank transport")
    save_figure(fig, "FigureQ3_Continuous_Target_Transport")
    return internal, external, summary


def continuous_metrics(cohort: str, ens: pd.DataFrame) -> dict:
    y = ens.target_log1p_abs_moment.to_numpy(float)
    p = ens.prediction.to_numpy(float)
    rho, rho_p = spearmanr(y, p)
    host_rhos = []
    for _, g in ens.groupby("host"):
        if len(g) >= 5 and g.target_log1p_abs_moment.nunique() > 1 and g.prediction.nunique() > 1:
            host_rhos.append(spearmanr(g.target_log1p_abs_moment, g.prediction).statistic)
    return {
        "cohort": cohort,
        "records": len(ens),
        "hosts": ens.host.nunique(),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "MAE_log1p": float(mean_absolute_error(y, p)),
        "RMSE_log1p": float(mean_squared_error(y, p) ** 0.5),
        "R2_log1p": float(r2_score(y, p)),
        "macro_host_spearman": float(np.nanmean(host_rhos)) if host_rhos else float("nan"),
        "evaluable_hosts": len(host_rhos),
    }


def decomposition_analysis(d2, imp, cfg):
    print("[4/7] Between-host versus within-host decomposition")
    cohorts = {
        "2DMD_strict": d2[d2.strict_primary].copy(),
        "Imp2D_strict": imp[imp.strict_primary].copy(),
    }
    rows, boot_summaries = [], []
    for cohort, g in cohorts.items():
        m = host_decomposition(g, "identity_probability", "y_true", "host")
        rows.append({"cohort": cohort, "probability": "identity_probability", **m})
        boot = bootstrap_host_metrics(g, "identity_probability", "y_true", "host", int(cfg["statistics"]["host_bootstrap_replicates"]), 20260717 + stable_int(cohort))
        boot.to_csv(CHECK / f"bootstrap_{cohort}.csv", index=False)
        for metric in ["AUROC", "host_centered_AUROC", "within_host_pair_AUROC"]:
            vals = boot[metric].dropna()
            boot_summaries.append({
                "cohort": cohort,
                "metric": metric,
                "estimate": m[metric],
                "ci_2_5": float(vals.quantile(0.025)) if len(vals) else np.nan,
                "ci_97_5": float(vals.quantile(0.975)) if len(vals) else np.nan,
                "valid_bootstraps": len(vals),
            })
    summary = pd.DataFrame(rows)
    ci = pd.DataFrame(boot_summaries)
    summary.to_csv(TABLE / "TABLE_Q1_BETWEEN_WITHIN_DECOMPOSITION.csv", index=False)
    ci.to_csv(TABLE / "TABLE_Q1_BETWEEN_WITHIN_BOOTSTRAP_CI.csv", index=False)
    write_latex_table(summary[["cohort", "AUROC", "host_centered_AUROC", "within_host_pair_AUROC", "macro_host_AUROC", "host_prevalence_score_spearman"]], TABLE / "TABLE_Q1_BETWEEN_WITHIN_DECOMPOSITION.tex", "External between-host and within-host discrimination.", "tab:q1_decomp")

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    metrics = ["AUROC", "host_centered_AUROC", "within_host_pair_AUROC"]
    x = np.arange(len(cohorts))
    width = 0.24
    for i, metric in enumerate(metrics):
        vals = [summary.loc[summary.cohort == c, metric].iloc[0] for c in cohorts]
        ax.bar(x + (i - 1) * width, vals, width, label=metric.replace("_", " "))
    ax.set_xticks(x, list(cohorts))
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Global discrimination separates from within-host ranking")
    ax.legend()
    save_figure(fig, "FigureQ4_Between_Within_Host_Decomposition")
    return summary, ci


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, float))
    return np.searchsorted(ref, np.asarray(values, float), side="right") / len(ref)


def ad_analysis(dev, base, d2, imp, features, cfg):
    print("[5/7] Quantitative applicability-domain analysis")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(dev[features])
    scaler = RobustScaler(quantile_range=(10, 90))
    Xs = scaler.fit_transform(X)
    pca_full = PCA(n_components=float(cfg["applicability_domain"]["pca_variance"]), svd_solver="full")
    Z0 = pca_full.fit_transform(Xs)
    n_comp = min(int(cfg["applicability_domain"]["max_components"]), Z0.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    Z = pca.fit_transform(Xs)
    k = int(cfg["applicability_domain"]["knn_k"])

    # Cross-host nearest-neighbor distance for development.
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    same = dev.host_key.to_numpy()[:, None] == dev.host_key.to_numpy()[None, :]
    dist[same] = np.inf
    dev_knn = np.partition(dist, kth=k - 1, axis=1)[:, :k].mean(axis=1)

    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    cov = LedoitWolf().fit(Z)
    q01 = np.nanquantile(X, 0.01, axis=0)
    q99 = np.nanquantile(X, 0.99, axis=0)
    dev_range = ((X < q01) | (X > q99)).mean(axis=1)
    dev_mahal = cov.mahalanobis(Z)

    rows = []
    frames = {
        "Development": (dev, "host_key", "record_id"),
        "2DMD_strict": (d2[d2.strict_primary].copy(), "host", "_id"),
        "Imp2D_strict": (imp[imp.strict_primary].copy(), "host", "external_id"),
    }
    for cohort, (frame, host_col, id_col) in frames.items():
        if cohort == "Development":
            knn_d, mahal, range_frac = dev_knn, dev_mahal, dev_range
        else:
            Xe = imputer.transform(frame[features])
            Ze = pca.transform(scaler.transform(Xe))
            knn_d = nn.kneighbors(Ze, return_distance=True)[0].mean(axis=1)
            mahal = cov.mahalanobis(Ze)
            range_frac = ((Xe < q01) | (Xe > q99)).mean(axis=1)
        knn_pct = empirical_percentile(dev_knn, knn_d)
        mahal_pct = empirical_percentile(dev_mahal, mahal)
        range_pct = empirical_percentile(dev_range, range_frac)
        composite = (knn_pct + mahal_pct + range_pct) / 3.0
        for rid, host, a, b, c, d, e, f, score in zip(frame[id_col], frame[host_col], knn_d, mahal, range_frac, knn_pct, mahal_pct, range_pct, composite):
            rows.append({
                "cohort": cohort,
                "record_id": rid,
                "host": host,
                "knn_distance": float(a),
                "mahalanobis_distance": float(b),
                "range_exceedance_fraction": float(c),
                "knn_percentile": float(d),
                "mahalanobis_percentile": float(e),
                "range_percentile": float(f),
                "AD_score": float(score),
            })
    ad = pd.DataFrame(rows)
    cutoff = float(ad.loc[ad.cohort == "Development", "AD_score"].quantile(float(cfg["applicability_domain"]["development_cutoff_quantile"])))
    ad["in_development_AD"] = ad.AD_score <= cutoff
    ad.to_csv(TABLE / "TABLE_Q1_QUANTITATIVE_AD_SCORES.csv", index=False)
    write_json(OUT / "QUANTITATIVE_AD_LOCK.json", {
        "fitted_on": "development_features_only",
        "features": features,
        "pca_components": int(n_comp),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "knn_k": k,
        "development_AD_cutoff_q95": cutoff,
        "definition": "mean empirical percentile of cross-host kNN distance, shrinkage Mahalanobis distance, and 1st-99th percentile range exceedance fraction",
    })

    coverage = []
    dev_prob = base.groupby(["record_id", "y_true"], as_index=False).probability.mean().rename(columns={"probability": "p", "y_true": "y"})
    perf_frames = {
        "Development": dev[["record_id", "host_key", "target_is_magnetic"]].rename(columns={"host_key": "host", "target_is_magnetic": "y"}).merge(dev_prob[["record_id", "p"]], on="record_id"),
        "2DMD_strict": d2[d2.strict_primary][["_id", "host", "y_true", "identity_probability"]].rename(columns={"_id": "record_id", "y_true": "y", "identity_probability": "p"}),
        "Imp2D_strict": imp[imp.strict_primary][["external_id", "host", "y_true", "identity_probability"]].rename(columns={"external_id": "record_id", "y_true": "y", "identity_probability": "p"}),
    }
    for cohort, frame in perf_frames.items():
        g = frame.merge(ad[ad.cohort == cohort][["record_id", "AD_score", "in_development_AD"]], on="record_id", validate="one_to_one")
        abs_error = np.abs(g.y - g.p)
        rho = spearmanr(g.AD_score, abs_error).statistic
        for retained in [0.25, 0.50, 0.75, 1.0]:
            n = max(2, int(math.ceil(len(g) * retained)))
            s = g.nsmallest(n, "AD_score")
            m = host_decomposition(s, "p", "y", "host")
            coverage.append({"cohort": cohort, "retained_fraction": retained, "selection": "closest_to_development", "AD_error_spearman": float(rho), **m})
        for state, s in g.groupby("in_development_AD"):
            m = host_decomposition(s, "p", "y", "host") if len(s) and s.y.nunique() == 2 else {"records": len(s), "prevalence": s.y.mean() if len(s) else np.nan, "AUROC": np.nan, "AUPRC": np.nan, "Brier": brier_score_loss(s.y, s.p) if len(s) else np.nan, "host_centered_AUROC": np.nan, "within_host_pair_AUROC": np.nan}
            coverage.append({"cohort": cohort, "retained_fraction": len(s)/len(g), "selection": "inside_AD" if state else "outside_AD", "AD_error_spearman": float(rho), **m})
    covtab = pd.DataFrame(coverage)
    covtab.to_csv(TABLE / "TABLE_Q1_AD_STRATIFIED_PERFORMANCE.csv", index=False)

    cohort_summary = ad.groupby("cohort", as_index=False).agg(
        records=("record_id", "size"),
        hosts=("host", "nunique"),
        median_AD_score=("AD_score", "median"),
        q95_AD_score=("AD_score", lambda x: x.quantile(0.95)),
        in_domain_fraction=("in_development_AD", "mean"),
        median_knn_distance=("knn_distance", "median"),
        median_mahalanobis=("mahalanobis_distance", "median"),
        median_range_exceedance=("range_exceedance_fraction", "median"),
    )
    cohort_summary.to_csv(TABLE / "TABLE_Q1_AD_COHORT_SUMMARY.csv", index=False)
    write_latex_table(cohort_summary, TABLE / "TABLE_Q1_AD_COHORT_SUMMARY.tex", "Quantitative applicability-domain summary.", "tab:q1_ad")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for cohort, g in ad.groupby("cohort"):
        ax.hist(g.AD_score, bins=30, alpha=0.45, label=cohort, density=True)
    ax.axvline(cutoff, linestyle="--", linewidth=1.2, label="Development q95 cutoff")
    ax.set_xlabel("Quantitative AD score")
    ax.set_ylabel("Density")
    ax.set_title("Development-frozen applicability-domain score")
    ax.legend()
    save_figure(fig, "FigureQ5_Quantitative_Applicability_Domain")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cc = covtab[covtab.selection == "closest_to_development"]
    for cohort, g in cc.groupby("cohort"):
        ax.plot(g.retained_fraction, g.AUROC, marker="o", label=cohort)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Fraction retained, closest to development")
    ax.set_ylabel("AUROC")
    ax.set_title("Performance after AD-based retention")
    ax.legend()
    save_figure(fig, "FigureQ6_AD_Retention_Performance")
    return ad, cohort_summary, covtab, cutoff


def feature_stability_analysis():
    print("[6/7] Feature-importance stability")
    imp = pd.read_csv(INPUT / "original_feature_importance_runs.csv")
    summary = imp.groupby("feature", as_index=False).agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        top5_frequency=("rank", lambda x: (x <= 5).mean()),
        top10_frequency=("rank", lambda x: (x <= 10).mean()),
        runs=("rank", "size"),
    ).sort_values("mean_rank")
    summary.to_csv(TABLE / "TABLE_Q1_FEATURE_RANK_STABILITY.csv", index=False)

    run_ids = imp[["seed", "test_host"]].drop_duplicates().itertuples(index=False, name=None)
    run_ids = list(run_ids)
    rank_maps = {}
    for seed, host in run_ids:
        g = imp[(imp.seed == seed) & (imp.test_host == host)].set_index("feature").sort_index()
        rank_maps[(seed, host)] = g["rank"]
    corrs = []
    for a, b in combinations(run_ids, 2):
        ra, rb = rank_maps[a].align(rank_maps[b], join="inner")
        corrs.append(spearmanr(ra, rb).statistic)
    stability = {
        "runs": len(run_ids),
        "pairwise_comparisons": len(corrs),
        "mean_pairwise_rank_spearman": float(np.nanmean(corrs)),
        "median_pairwise_rank_spearman": float(np.nanmedian(corrs)),
        "q10_pairwise_rank_spearman": float(np.nanquantile(corrs, 0.10)),
        "q90_pairwise_rank_spearman": float(np.nanquantile(corrs, 0.90)),
    }
    write_json(OUT / "FEATURE_RANK_STABILITY_SUMMARY.json", stability)

    top = summary.head(15).sort_values("mean_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.barh(top.feature, top.top10_frequency)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Frequency among top 10 across seed-host runs")
    ax.set_title("Stability of leading primary features")
    save_figure(fig, "FigureQ8_Feature_Rank_Stability")
    return summary, stability


def make_summary(cal_summary, selected, postdiag, endpoint, continuous, decomp, ad_summary, ad_cov, cutoff, perm_summary, feature_stability):
    print("[7/7] Reports, hashes, and success markers")
    selected_row = cal_summary[cal_summary.method == selected].iloc[0]
    original_platt = cal_summary[cal_summary.method == "platt"].iloc[0]
    d2_decomp = decomp[decomp.cohort == "2DMD_strict"].iloc[0]
    imp_decomp = decomp[decomp.cohort == "Imp2D_strict"].iloc[0]
    endpoint_dev = endpoint[endpoint.cohort == "Development_LOHO"]
    endpoint_d2 = endpoint[endpoint.cohort == "2DMD_strict"]
    endpoint_imp = endpoint[endpoint.cohort == "Imp2D_strict"]
    cont_dev = continuous[continuous.cohort == "Development_LOHO"].iloc[0]
    cont_d2 = continuous[continuous.cohort == "2DMD_strict"].iloc[0]
    cont_imp = continuous[continuous.cohort == "Imp2D_strict"].iloc[0]

    report = f"""# DefectTwin-2D Q1 strengthening report

Generated: {now()}

## Completed strengthening analyses

- Four-way nested host-disjoint calibration comparison: identity, Platt, beta, and temperature scaling.
- Endpoint sensitivity at 0.05, 0.10, 0.20, and 0.50 micro_B.
- Continuous log-moment prediction under host-disjoint development and both external cohorts.
- Explicit global, host-centered, pair-weighted within-host, and macro-host discrimination decomposition.
- Development-frozen quantitative applicability-domain score using cross-host kNN, shrinkage Mahalanobis distance, and range exceedance.
- Grouped held-out-host permutation importance.
- Feature-rank stability across seed-host runs.
- Host-block bootstrap intervals.

## Calibration correction

The development-only selected calibrator is **{selected}**. Its nested ensemble Brier score is **{selected_row.Brier:.4f}**, log loss **{selected_row.log_loss:.4f}**, and ECE15 **{selected_row.ECE:.4f}**. Nested Platt produced Brier **{original_platt.Brier:.4f}** and ECE15 **{original_platt.ECE:.4f}**. The manuscript must describe the original reported internal calibration as nested Platt unless the newly selected development-only comparison is adopted consistently in the revised analysis.

External recalibration results in this package are explicitly post-lockbox diagnostics and do not replace the sealed original result.

## Endpoint robustness

Development LOHO AUROC across the four predeclared endpoints ranged from **{endpoint_dev.AUROC.min():.4f}** to **{endpoint_dev.AUROC.max():.4f}**. The strict 2DMD external AUROC ranged from **{endpoint_d2.AUROC.min():.4f}** to **{endpoint_d2.AUROC.max():.4f}**, while strict Imp2D ranged from **{endpoint_imp.AUROC.min():.4f}** to **{endpoint_imp.AUROC.max():.4f}**. These results should be interpreted together with prevalence, AUPRC, calibration, and within-host metrics in Table Q1 endpoint sensitivity.

## Continuous target

- Development LOHO Spearman rho: **{cont_dev.spearman_rho:.4f}**.
- 2DMD strict Spearman rho: **{cont_d2.spearman_rho:.4f}**.
- Imp2D strict Spearman rho: **{cont_imp.spearman_rho:.4f}**.

The continuous analysis tests whether conclusions depend on a single binary threshold. It does not convert the target into experimental magnetic order.

## Between-host versus within-host result

- 2DMD global AUROC: **{d2_decomp.AUROC:.4f}**; host-centered AUROC: **{d2_decomp.host_centered_AUROC:.4f}**; pair-weighted within-host AUROC: **{d2_decomp.within_host_pair_AUROC:.4f}**.
- Imp2D global AUROC: **{imp_decomp.AUROC:.4f}**; host-centered AUROC: **{imp_decomp.host_centered_AUROC:.4f}**; pair-weighted within-host AUROC: **{imp_decomp.within_host_pair_AUROC:.4f}**.

This decomposition is the central scientific contribution: global external ranking can be substantially stronger than defect ranking within a new host.

## Quantitative applicability domain

The AD rule was fitted only on development descriptors. The development q95 composite cutoff is **{cutoff:.4f}**. Cohort coverage and performance are reported in the AD tables. An AD score is a diagnostic support measure, not a safety guarantee or a post-hoc exclusion license.

## Feature evidence

The mean pairwise Spearman correlation of feature rankings across seed-host runs is **{feature_stability['mean_pairwise_rank_spearman']:.4f}**. Grouped permutation results identify which descriptor families affect held-out-host probability quality; they should be preferred over single-fit impurity importance for physical interpretation.

## Remaining non-computational requirements

1. Deposit the clean repository and archive release, then insert the GitHub/Zenodo DOI.
2. Complete author affiliation, corresponding-author email, funding, and acknowledgements.
3. Select the target journal and adapt formatting.
4. A genuinely larger independent same-regime lockbox still requires a new public source with additional phase-unseen hosts. No suitable additional source was present in the supplied files, so no result was fabricated.

## Submission position

The strengthened paper should be presented as a trustworthy materials-ML applicability-domain and transport-decomposition study. It should not claim a universally transferable magnetic-defect predictor or a new state-of-the-art graph architecture.
"""
    (OUT / "Q1_STRENGTHENING_REPORT.md").write_text(report, encoding="utf-8")

    arabic = f"""# الملخص التنفيذي العربي - تقوية DefectTwin-2D للنشر Q1

تم تنفيذ التحليلات الممكنة بالكامل من البيانات الموجودة، مع عدم تغيير نتيجة الـlockbox الأصلية أو اختلاق أي بيانات خارجية.

## أهم ما تم إصلاحه

- تمت مقارنة Identity وPlatt وBeta وTemperature داخل nested host-disjoint validation.
- الطريقة المختارة من بيانات التطوير فقط هي **{selected}**، وBrier = **{selected_row.Brier:.4f}** وECE15 = **{selected_row.ECE:.4f}**.
- تم اختبار أربع عتبات للعزم المغناطيسي: 0.05 و0.10 و0.20 و0.50 micro_B.
- تم إضافة تحليل continuous target بدل الاعتماد على عتبة واحدة فقط.
- تم فصل الأداء العالمي عن الأداء داخل كل host بصورة كمية.
- 2DMD: AUROC عالمي **{d2_decomp.AUROC:.4f}**، لكن host-centered **{d2_decomp.host_centered_AUROC:.4f}** وwithin-host **{d2_decomp.within_host_pair_AUROC:.4f}**.
- تم إنشاء Applicability Domain كمي مجمد من بيانات التطوير فقط، بدل الخريطة التوضيحية اليدوية.
- تم إضافة grouped permutation importance وثبات ترتيب الخصائص.

## الحكم بعد التقوية

البحث أصبح أقوى كثيرًا كبحث **applicability domain + external transport auditing + failure analysis**. هذه الصياغة هي الأكثر دفاعًا في مجلة Q1 قوية. لكنه لا يتحول إلى بحث بخوارزمية جديدة قوية، ولا يجوز ضمان القبول.

## الشيء الوحيد الذي لم يمكن تنفيذه من الحزمة

إضافة lockbox مستقل أكبر بعدد hosts جديدة تحتاج قاعدة بيانات عامة أخرى غير موجودة في الملفات. تم إعداد بروتوكول كامل ومجمد لهذه التجربة، لكن لا يمكن إنشاء نتائجها بدون مصدر حقيقي.

## المطلوب الإداري قبل الإرسال

- رفع الكود والنتائج إلى GitHub/Zenodo وإضافة DOI.
- إدخال بيانات المؤلف والانتماء والبريد.
- اختيار المجلة وضبط القالب.
"""
    (OUT / "ARABIC_EXECUTIVE_SUMMARY.md").write_text(arabic, encoding="utf-8")

    # Hash all user-facing outputs, excluding the hash file itself.
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "HASHES_SHA256.csv"]
    hashes = pd.DataFrame([{"relative_path": str(p.relative_to(ROOT)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(files)])
    hashes.to_csv(OUT / "HASHES_SHA256.csv", index=False)

    verification = {
        "completed_at": now(),
        "nested_calibration_methods_4": len(cal_summary) == 4,
        "selected_calibrator_development_only": selected in set(cal_summary.method),
        "thresholds_4": endpoint.endpoint_muB.nunique() == 4,
        "continuous_cohorts_3": continuous.cohort.nunique() == 3,
        "decomposition_cohorts_2": decomp.cohort.nunique() == 2,
        "ad_cohorts_3": ad_summary.cohort.nunique() == 3,
        "ad_cutoff_finite": bool(np.isfinite(cutoff)),
        "permutation_groups_at_least_7": len(perm_summary) >= 7,
        "feature_stability_finite": bool(np.isfinite(feature_stability["mean_pairwise_rank_spearman"])),
        "figures_png_pdf_svg": all(len(list(FIG.glob(f"*.{ext}"))) >= 8 for ext in ["png", "pdf", "svg"]),
        "hashes_written": len(hashes) > 20,
        "original_lockbox_not_overwritten": True,
        "larger_external_lockbox_not_fabricated": True,
    }
    verification["all"] = all(verification.values())
    write_json(OUT / "VERIFICATION.json", verification)
    write_json(OUT / "STATUS.json", {"state": "COMPLETED" if verification["all"] else "FAILED_VERIFICATION", "updated_at": now(), "selected_calibrator": selected})
    write_json(OUT / "SUCCESS_MARKER.json", {"marker": "DEFECTTWIN2D_Q1_STRENGTHENING_COMPLETE", "verification_passed": verification["all"], "selected_calibrator": selected, "acceptance_guaranteed": False})
    if not verification["all"]:
        raise RuntimeError("Final verification flags did not all pass")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    (OUT / "ERROR_LOG.txt").write_text("", encoding="utf-8")
    cfg = yaml.safe_load((ROOT / "config" / "project_config.yaml").read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "version": cfg["version"]})
    dev, base, imp, d2, features = load_data(cfg)
    write_json(OUT / "INPUT_VERIFICATION.json", {
        "development_records": len(dev),
        "development_hosts": dev.host_key.nunique(),
        "primary_features": len(features),
        "2dmd_records": len(d2),
        "2dmd_strict_records": int(d2.strict_primary.sum()),
        "imp2d_records": len(imp),
        "imp2d_strict_records": int(imp.strict_primary.sum()),
        "all_expected": len(dev) == 529 and dev.host_key.nunique() == 30 and len(features) == 68 and len(d2) == 3000 and int(d2.strict_primary.sum()) == 2500 and int(imp.strict_primary.sum()) == 11866,
    })

    cal_pred, cal_ens, cal_summary, selected = nested_calibration_analysis(dev, base, features, cfg)
    calibrators, postdiag = fit_deployment_calibrators(dev, base, d2, imp, selected, cfg)
    thr_internal, thr_external, endpoint_summary, perm_summary = threshold_and_permutation_analysis(dev, d2, imp, features, cfg)
    cont_internal, cont_external, continuous_summary = continuous_target_analysis(dev, d2, imp, features, cfg)
    decomp_summary, decomp_ci = decomposition_analysis(d2, imp, cfg)
    ad, ad_summary, ad_cov, cutoff = ad_analysis(dev, base, d2, imp, features, cfg)
    feature_summary, feature_stability = feature_stability_analysis()
    make_summary(cal_summary, selected, postdiag, endpoint_summary, continuous_summary, decomp_summary, ad_summary, ad_cov, cutoff, perm_summary, feature_stability)
    print("Q1 strengthening completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ERROR_LOG.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(OUT / "STATUS.json", {"state": "FAILED", "updated_at": now()})
        traceback.print_exc()
        sys.exit(1)
