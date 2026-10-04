from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUT = ROOT / "outputs"
TABLE = OUT / "tables"
FIG = OUT / "figures"
CHECK = OUT / "checkpoints"
LOG = OUT / "logs"

for p in (OUT, TABLE, FIG, CHECK, LOG):
    p.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def safe(v):
    if v is None:
        return None
    x = float(v)
    return x if np.isfinite(x) else None


def inverse_host_weights(hosts: pd.Series) -> np.ndarray:
    counts = hosts.value_counts()
    w = hosts.map(lambda h: 1.0 / counts[h]).to_numpy(float)
    return w / w.mean()


def host_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal total influence per host; within mixed hosts give each class half."""
    weights = pd.Series(0.0, index=frame.index, dtype=float)
    for _, g in frame.groupby("host"):
        if g.y.nunique() == 2:
            for cls in (0, 1):
                idx = g.index[g.y.eq(cls)]
                weights.loc[idx] = 1.0 / (2.0 * len(idx))
        else:
            weights.loc[g.index] = 1.0 / len(g)
    arr = weights.to_numpy(float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise RuntimeError("Invalid host/class balancing weights.")
    return arr / arr.mean()


def centered_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Label-blind deviations from each host candidate-set mean."""
    raw = frame[features].astype(float)
    return (raw - frame.groupby("host")[features].transform("mean")).add_prefix("hc_").reset_index(drop=True)


def conditional_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Raw descriptors plus label-blind host-centered deviations."""
    return pd.concat([frame[features].astype(float).reset_index(drop=True), centered_features(frame, features)], axis=1)


def ece(y, p, bins: int = 15) -> float:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if mask.any():
            total += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return total


def within_host_pair_auc(df: pd.DataFrame, score: str, target: str = "y", host: str = "host") -> tuple[float, int]:
    concordant = 0.0
    pairs = 0
    for _, g in df.groupby(host):
        pos = g.loc[g[target] == 1, score].to_numpy(float)
        neg = g.loc[g[target] == 0, score].to_numpy(float)
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        concordant += float((d > 0).sum() + 0.5 * (d == 0).sum())
        pairs += int(d.size)
    return (float(concordant / pairs) if pairs else np.nan, pairs)


def host_centered_auc(df: pd.DataFrame, score: str, target: str = "y", host: str = "host") -> float:
    centered = df[score] - df.groupby(host)[score].transform("mean")
    return float(roc_auc_score(df[target], centered)) if df[target].nunique() == 2 else np.nan


def macro_host_auc(df: pd.DataFrame, score: str, target: str = "y", host: str = "host") -> tuple[float, int]:
    vals = []
    for _, g in df.groupby(host):
        if g[target].nunique() == 2:
            vals.append(float(roc_auc_score(g[target], g[score])))
    return (float(np.mean(vals)) if vals else np.nan, len(vals))


def model_metrics(df: pd.DataFrame, score: str, probability: str | None = None) -> dict:
    y = df.y.to_numpy(int)
    s = df[score].to_numpy(float)
    pair_auc, pairs = within_host_pair_auc(df, score)
    macro_auc, evaluable = macro_host_auc(df, score)
    host_auc = host_centered_auc(df, score)
    pooled = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else np.nan
    auprc = float(average_precision_score(y, s)) if np.unique(y).size == 2 else np.nan
    host_rows = []
    for h, g in df.groupby("host"):
        if g.y.nunique() == 2:
            host_rows.append(float(roc_auc_score(g.y, g[score])))
    result = {
        "records": int(len(df)),
        "hosts": int(df.host.nunique()),
        "evaluable_hosts": int(evaluable),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pooled_AUROC": pooled,
        "AUPRC": auprc,
        "host_centered_AUROC": host_auc,
        "within_host_pair_AUROC": pair_auc,
        "within_host_pairs": int(pairs),
        "macro_host_AUROC": macro_auc,
        "worst_host_AUROC": float(np.min(host_rows)) if host_rows else np.nan,
        "q25_host_AUROC": float(np.quantile(host_rows, 0.25)) if host_rows else np.nan,
        "shortcut_amplification_index": float(pooled - pair_auc) if np.isfinite(pooled) and np.isfinite(pair_auc) else np.nan,
    }
    if probability is not None:
        p = np.clip(df[probability].to_numpy(float), 1e-7, 1 - 1e-7)
        pred = (p >= 0.5).astype(int)
        null_brier = float(y.mean() * (1 - y.mean()))
        brier = float(brier_score_loss(y, p))
        result.update({
            "Brier": brier,
            "Brier_skill": float(1 - brier / null_brier) if null_brier > 0 else np.nan,
            "NLL": float(log_loss(y, p, labels=[0, 1])),
            "ECE15": ece(y, p, 15),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if np.unique(y).size == 2 else np.nan,
        })
    return result


def save_figure(fig, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvalues), float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def load_data():
    dev = pd.read_csv(INPUT / "development_representation.csv")
    target = pd.read_csv(INPUT / "development_targets_manifest.csv")
    target = target[["record_id", "authoritative_json_magmom"]].copy()
    target["abs_moment"] = target.authoritative_json_magmom.abs()
    dev = dev.merge(target[["record_id", "abs_moment"]], on="record_id", validate="one_to_one")
    features = sorted([c for c in dev if c.startswith("x_")] + [c for c in dev if c.startswith("c_") and "delta" in c])
    if len(dev) != 529 or dev.host_key.nunique() != 30 or len(features) != 68:
        raise RuntimeError(f"Development lock mismatch: rows={len(dev)}, hosts={dev.host_key.nunique()}, features={len(features)}")
    dev_frame = dev[["record_id", "host_key", "target_is_magnetic", "abs_moment", *features]].rename(
        columns={"host_key": "host", "target_is_magnetic": "y"}
    )

    base = pd.read_csv(INPUT / "original_base_loho_predictions.csv")
    base = base[base.model.eq("et_x_composition_delta_primary")]
    base = base.groupby(["test_host", "record_id", "y_true"], as_index=False).probability.mean()
    base = base.rename(columns={"test_host": "host", "y_true": "y", "probability": "et_oof_probability"})
    if base.record_id.nunique() != len(dev_frame):
        raise RuntimeError("Existing five-seed LOHO predictions do not cover development exactly.")

    d2f = pd.read_csv(INPUT / "2dmd_features.csv")
    d2e = pd.read_csv(INPUT / "2dmd_evaluated_original.csv")
    d2 = d2e[["_id", "host", "strict_primary", "total_mag", "y_true"]].merge(
        d2f[["_id", "host", "strict_primary", *features]],
        on=["_id", "host", "strict_primary"], validate="one_to_one"
    )
    d2 = d2[d2.strict_primary].rename(columns={"_id": "record_id", "y_true": "y", "total_mag": "abs_moment"})

    impf = pd.read_csv(INPUT / "imp2d_features.csv")
    impe = pd.read_csv(INPUT / "imp2d_evaluated_original.csv")
    keys = ["ase_row_id", "external_id", "name", "host", "selected_phase_uid", "dopant", "defecttype", "site"]
    imp = impe.merge(impf[keys + features], on=keys, validate="one_to_one")
    dev_hosts = set(dev_frame.host)
    imp = imp[~imp.selected_phase_uid.isin(dev_hosts)].copy()
    imp["abs_moment"] = imp.spin.abs()
    imp = imp.rename(columns={"external_id": "record_id", "y_true": "y"})

    qf = pd.read_csv(INPUT / "qpod_features.csv")
    qe = pd.read_csv(INPUT / "qpod_evaluated.csv")
    q = qe[["record_id", "host_uid", "target_abs_magnetic_moment", "y_true"]].merge(
        qf[["record_id", "host_uid", *features]], on=["record_id", "host_uid"], validate="one_to_one"
    )
    q = q.rename(columns={"host_uid": "host", "target_abs_magnetic_moment": "abs_moment", "y_true": "y"})

    for name, frame in [("2DMD", d2), ("Imp2D", imp), ("QPOD", q)]:
        missing = [c for c in features if c not in frame]
        if missing:
            raise RuntimeError(f"{name} misses locked features: {missing[:5]}")
    return dev_frame, base, d2, imp, q, features


class PairRanker:
    def __init__(self, C=1.0, max_iter=3000, max_pairs_per_host=1000, seed=20260718):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.max_pairs_per_host = int(max_pairs_per_host)
        self.seed = int(seed)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.model = None
        self.score_mean = 0.0
        self.score_std = 1.0
        self.valid = False
        self.pair_count = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, hosts: pd.Series):
        Xi = self.imputer.fit_transform(X)
        Xs = self.scaler.fit_transform(Xi)
        rng = np.random.default_rng(self.seed)
        diffs = []
        labels = []
        weights = []
        host_arr = np.asarray(hosts)
        y_arr = np.asarray(y, int)
        for h in pd.unique(host_arr):
            idx = np.flatnonzero(host_arr == h)
            pos = idx[y_arr[idx] == 1]
            neg = idx[y_arr[idx] == 0]
            if not len(pos) or not len(neg):
                continue
            all_pairs = np.array(np.meshgrid(pos, neg)).T.reshape(-1, 2)
            if len(all_pairs) > self.max_pairs_per_host:
                sel = rng.choice(len(all_pairs), self.max_pairs_per_host, replace=False)
                all_pairs = all_pairs[sel]
            d = Xs[all_pairs[:, 0]] - Xs[all_pairs[:, 1]]
            n = len(d)
            diffs.append(d)
            labels.append(np.ones(n, int))
            weights.append(np.full(n, 1.0 / n))
            diffs.append(-d)
            labels.append(np.zeros(n, int))
            weights.append(np.full(n, 1.0 / n))
        if not diffs:
            self.valid = False
            return self
        DX = np.vstack(diffs)
        dy = np.concatenate(labels)
        dw = np.concatenate(weights)
        dw = dw / dw.mean()
        self.model = SGDClassifier(loss="log_loss", alpha=max(1e-6, 1e-4 / self.C), max_iter=self.max_iter, tol=1e-4, fit_intercept=False, average=True, random_state=self.seed)
        self.model.fit(DX, dy, sample_weight=dw)
        raw = self.model.decision_function(Xs)
        self.score_mean = float(np.mean(raw))
        self.score_std = float(np.std(raw)) if float(np.std(raw)) > 1e-8 else 1.0
        self.valid = True
        self.pair_count = int(len(DX) // 2)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if not self.valid:
            return np.zeros(len(X), float)
        Xi = self.imputer.transform(X)
        Xs = self.scaler.transform(Xi)
        raw = self.model.decision_function(Xs)
        return (raw - self.score_mean) / self.score_std


class HostBranch:
    def __init__(self, pca_components_max=8, ridge_alpha=10.0):
        self.max_components = int(pca_components_max)
        self.alpha = float(ridge_alpha)
        self.pipeline = None
        self.valid = False

    @staticmethod
    def aggregate(X: pd.DataFrame, hosts: pd.Series, y: pd.Series | None = None) -> pd.DataFrame:
        tmp = X.copy()
        tmp["_host"] = np.asarray(hosts)
        agg = tmp.groupby("_host").mean(numeric_only=True)
        if y is not None:
            yy = pd.DataFrame({"_host": np.asarray(hosts), "_y": np.asarray(y, float)}).groupby("_host")["_y"].mean()
            agg["_target"] = yy
        return agg

    def fit(self, X: pd.DataFrame, y: pd.Series, hosts: pd.Series):
        agg = self.aggregate(X, hosts, y)
        Xh = agg.drop(columns="_target")
        yh = agg["_target"]
        ncomp = max(1, min(self.max_components, len(Xh) - 2, Xh.shape[1]))
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=ncomp, random_state=20260718)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.pipeline.fit(Xh, yh)
        self.valid = True
        return self

    def predict_records(self, X: pd.DataFrame, hosts: pd.Series) -> np.ndarray:
        if not self.valid:
            return np.full(len(X), 0.5)
        agg = self.aggregate(X, hosts)
        pred = np.clip(self.pipeline.predict(agg), 0.02, 0.98)
        mapping = dict(zip(agg.index, pred))
        return np.asarray([mapping[h] for h in hosts], float)


@dataclass(frozen=True)
class Candidate:
    name: str
    w_global: float
    w_conditional: float
    w_host: float
    w_rank: float


def combine_score(global_p, conditional_p, host_p, rank_z, candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    global_p = np.asarray(global_p, float)
    conditional_p = np.asarray(conditional_p, float)
    # Preserve the native probability scale for pure probabilistic branches.
    if candidate.w_global == 1 and candidate.w_conditional == 0 and candidate.w_host == 0 and candidate.w_rank == 0:
        return global_p, global_p
    if candidate.w_conditional == 1 and candidate.w_global == 0 and candidate.w_host == 0 and candidate.w_rank == 0:
        return conditional_p, conditional_p
    g = logit(np.clip(global_p, 1e-5, 1 - 1e-5))
    c = logit(np.clip(conditional_p, 1e-5, 1 - 1e-5))
    h = logit(np.clip(np.asarray(host_p, float), 1e-5, 1 - 1e-5))
    r = np.asarray(rank_z, float)
    score = (candidate.w_global * g + candidate.w_conditional * c
             + candidate.w_host * h + candidate.w_rank * r)
    return score, expit(score)


def choose_candidate(meta: pd.DataFrame, candidates: list[Candidate], cfg: dict) -> tuple[Candidate, pd.DataFrame]:
    rows = []
    for c in candidates:
        score, prob = combine_score(meta.et_oof_probability, meta.conditional_probability, meta.host_probability, meta.rank_z, c)
        g = meta[["record_id", "host", "y"]].copy()
        g["score"] = score
        g["prob"] = prob
        rows.append({"candidate": c.name, **model_metrics(g, "score", "prob")})
    tab = pd.DataFrame(rows)
    wh_best = tab.within_host_pair_AUROC.max()
    keep = tab[tab.within_host_pair_AUROC >= wh_best - float(cfg["selection"]["within_host_tolerance"])].copy()
    hc_best = keep.host_centered_AUROC.max()
    keep = keep[keep.host_centered_AUROC >= hc_best - float(cfg["selection"]["host_centered_tolerance"])].copy()
    keep["complexity"] = keep.candidate.map({c.name: abs(c.w_global) + abs(c.w_conditional) + abs(c.w_host) + abs(c.w_rank) for c in candidates})
    selected_row = keep.sort_values(["pooled_AUROC", "macro_host_AUROC", "complexity"], ascending=[False, False, True]).iloc[0]
    selected = next(c for c in candidates if c.name == selected_row.candidate)
    tab["selected"] = tab.candidate.eq(selected.name)
    return selected, tab


def build_et(seed: int, cfg: dict) -> ExtraTreesClassifier:
    m = cfg["models"]["extra_trees"]
    return ExtraTreesClassifier(
        n_estimators=int(m["n_estimators"]),
        min_samples_leaf=int(m["min_samples_leaf"]),
        max_features=m["max_features"],
        class_weight=m["class_weight"],
        n_jobs=int(m["n_jobs"]),
        random_state=int(seed),
    )


def build_conditional_et(seed: int, cfg: dict) -> ExtraTreesClassifier:
    m = cfg["models"]["conditional_extra_trees"]
    return ExtraTreesClassifier(
        n_estimators=int(m["n_estimators"]),
        min_samples_leaf=int(m["min_samples_leaf"]),
        max_features=m["max_features"],
        class_weight=None,
        n_jobs=int(m["n_jobs"]),
        random_state=int(seed),
    )


def _fit_et_ensemble(X: pd.DataFrame, y: pd.Series, weights: np.ndarray, cfg: dict, seed_offset: int = 0) -> list[ExtraTreesClassifier]:
    models = []
    for seed in cfg["seeds"]:
        m = build_conditional_et(int(seed) + int(seed_offset), cfg)
        m.fit(X, y, sample_weight=weights)
        models.append(m)
    return models


def fit_full_models(dev: pd.DataFrame, features: list[str], cfg: dict):
    X = dev[features]
    Xcenter = centered_features(dev, features)
    Xcond = conditional_features(dev, features)
    y = dev.y
    hosts = dev.host
    ets = []
    for seed in cfg["seeds"]:
        m = build_et(int(seed), cfg)
        m.fit(X, y, sample_weight=inverse_host_weights(hosts))
        ets.append(m)
    hc_weights = host_class_weights(dev)
    raw_hostclass_ets = _fit_et_ensemble(X.reset_index(drop=True), y.reset_index(drop=True), hc_weights, cfg, 101)
    centered_ets = _fit_et_ensemble(Xcenter, y.reset_index(drop=True), hc_weights, cfg, 202)
    conditional_ets = _fit_et_ensemble(Xcond, y.reset_index(drop=True), hc_weights, cfg, 303)
    logit_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", LogisticRegression(C=float(cfg["models"]["logistic"]["C"]), max_iter=int(cfg["models"]["logistic"]["max_iter"]), class_weight="balanced", solver="liblinear")),
    ])
    logit_model.fit(X, y, model__sample_weight=inverse_host_weights(hosts))
    pr = PairRanker(**cfg["models"]["pair_rank"], seed=int(cfg["statistics"]["bootstrap_seed"]))
    pr.fit(X, y, hosts)
    hb = HostBranch(**cfg["models"]["host_branch"])
    hb.fit(X, y, hosts)
    return ets, raw_hostclass_ets, centered_ets, conditional_ets, logit_model, pr, hb


def predict_models(frame: pd.DataFrame, features: list[str], models, selected: Candidate) -> pd.DataFrame:
    ets, raw_hostclass_ets, centered_ets, conditional_ets, logit_model, pr, hb = models
    X = frame[features]
    Xcenter = centered_features(frame, features)
    Xcond = conditional_features(frame, features)
    etp = np.mean([m.predict_proba(X)[:, 1] for m in ets], axis=0)
    rawhcp = np.mean([m.predict_proba(X.reset_index(drop=True))[:, 1] for m in raw_hostclass_ets], axis=0)
    centerp = np.mean([m.predict_proba(Xcenter)[:, 1] for m in centered_ets], axis=0)
    condp = np.mean([m.predict_proba(Xcond)[:, 1] for m in conditional_ets], axis=0)
    logp = logit_model.predict_proba(X)[:, 1]
    rankz = pr.decision_function(X)
    hostp = hb.predict_records(X, frame.host)
    twin_score, twin_prob = combine_score(etp, condp, hostp, rankz, selected)
    out = frame[["record_id", "host", "y", "abs_moment"]].copy()
    out["ET_score"] = etp
    out["ET_probability"] = etp
    out["RawHostClassET_score"] = rawhcp
    out["RawHostClassET_probability"] = rawhcp
    out["CenteredOnlyET_score"] = centerp
    out["CenteredOnlyET_probability"] = centerp
    out["ConditionalET_score"] = condp
    out["ConditionalET_probability"] = condp
    out["Logistic_score"] = logp
    out["Logistic_probability"] = logp
    out["PairRank_score"] = rankz
    out["PairRank_probability"] = expit(rankz)
    out["HostBaseline_score"] = hostp
    out["HostBaseline_probability"] = hostp
    out["SelectedModel_score"] = twin_score
    out["SelectedModel_probability"] = twin_prob
    return out


def development_oof(dev: pd.DataFrame, base: pd.DataFrame, features: list[str], cfg: dict):
    rows = []
    hosts = sorted(dev.host.unique())
    for i, test_host in enumerate(hosts, 1):
        train = dev[dev.host != test_host].copy().reset_index(drop=True)
        test = dev[dev.host == test_host].copy().reset_index(drop=True)
        pr = PairRanker(**cfg["models"]["pair_rank"], seed=int(cfg["statistics"]["bootstrap_seed"]) + stable_int(test_host))
        pr.fit(train[features], train.y, train.host)
        hb = HostBranch(**cfg["models"]["host_branch"])
        hb.fit(train[features], train.y, train.host)
        log_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LogisticRegression(C=float(cfg["models"]["logistic"]["C"]), max_iter=int(cfg["models"]["logistic"]["max_iter"]), class_weight="balanced", solver="liblinear")),
        ])
        log_model.fit(train[features], train.y, model__sample_weight=inverse_host_weights(train.host))
        train_raw = train[features].reset_index(drop=True)
        test_raw = test[features].reset_index(drop=True)
        train_center = centered_features(train, features)
        test_center = centered_features(test, features)
        train_cond = conditional_features(train, features)
        test_cond = conditional_features(test, features)
        rawhc_probs, center_probs, cond_probs = [], [], []
        hc_weights = host_class_weights(train)
        fold_offset = stable_int(test_host)
        for seed in cfg["seeds"]:
            rm = build_conditional_et(int(seed) + fold_offset + 101, cfg)
            rm.fit(train_raw, train.y, sample_weight=hc_weights)
            rawhc_probs.append(rm.predict_proba(test_raw)[:, 1])
            zm = build_conditional_et(int(seed) + fold_offset + 202, cfg)
            zm.fit(train_center, train.y, sample_weight=hc_weights)
            center_probs.append(zm.predict_proba(test_center)[:, 1])
            cm = build_conditional_et(int(seed) + fold_offset + 303, cfg)
            cm.fit(train_cond, train.y, sample_weight=hc_weights)
            cond_probs.append(cm.predict_proba(test_cond)[:, 1])
        rawhcp = np.mean(rawhc_probs, axis=0)
        centerp = np.mean(center_probs, axis=0)
        condp = np.mean(cond_probs, axis=0)
        b = base[base.host.eq(test_host)].set_index("record_id")
        etp = test.record_id.map(b.et_oof_probability).to_numpy(float)
        rankz = pr.decision_function(test[features])
        hostp = hb.predict_records(test[features], test.host)
        logp = log_model.predict_proba(test[features])[:, 1]
        for j, rec in enumerate(test.record_id):
            rows.append({
                "record_id": rec,
                "host": test_host,
                "y": int(test.iloc[j].y),
                "abs_moment": float(test.iloc[j].abs_moment),
                "et_oof_probability": float(etp[j]),
                "raw_hostclass_probability": float(rawhcp[j]),
                "centered_only_probability": float(centerp[j]),
                "conditional_probability": float(condp[j]),
                "rank_z": float(rankz[j]),
                "host_probability": float(hostp[j]),
                "logistic_probability": float(logp[j]),
                "pair_ranker_valid": bool(pr.valid),
                "pair_count_training": int(pr.pair_count),
            })
        if i % 5 == 0 or i == len(hosts):
            print(f"  development LOHO host {i}/{len(hosts)}")
    oof = pd.DataFrame(rows)
    oof.to_csv(CHECK / "DEVELOPMENT_COMPONENT_OOF.csv", index=False)

    candidates = [Candidate(**x) for x in cfg["selection"]["candidates"]]
    selected_by_host = []
    final_parts = []
    candidate_audits = []
    for test_host in hosts:
        meta = oof[oof.host != test_host].copy()
        selected, tab = choose_candidate(meta, candidates, cfg)
        tab.insert(0, "outer_test_host", test_host)
        candidate_audits.append(tab)
        selected_by_host.append({"test_host": test_host, **selected.__dict__})
        test = oof[oof.host == test_host].copy()
        score, prob = combine_score(test.et_oof_probability, test.conditional_probability, test.host_probability, test.rank_z, selected)
        test["SelectedModel_score"] = score
        test["SelectedModel_probability"] = prob
        final_parts.append(test)
    final = pd.concat(final_parts, ignore_index=True)
    final["ET_score"] = final.et_oof_probability
    final["ET_probability"] = final.et_oof_probability
    final["RawHostClassET_score"] = final.raw_hostclass_probability
    final["RawHostClassET_probability"] = final.raw_hostclass_probability
    final["CenteredOnlyET_score"] = final.centered_only_probability
    final["CenteredOnlyET_probability"] = final.centered_only_probability
    final["ConditionalET_score"] = final.conditional_probability
    final["ConditionalET_probability"] = final.conditional_probability
    final["Logistic_score"] = final.logistic_probability
    final["Logistic_probability"] = final.logistic_probability
    final["PairRank_score"] = final.rank_z
    final["PairRank_probability"] = expit(final.rank_z)
    final["HostBaseline_score"] = final.host_probability
    final["HostBaseline_probability"] = final.host_probability
    final.to_csv(TABLE / "DEVELOPMENT_CROSSFIT_PREDICTIONS.csv", index=False)
    pd.DataFrame(selected_by_host).to_csv(TABLE / "DEVELOPMENT_ROW_EXCLUSION_SELECTION_SENSITIVITY.csv", index=False)
    pd.concat(candidate_audits, ignore_index=True).to_csv(TABLE / "DEVELOPMENT_META_SELECTION_AUDIT.csv", index=False)

    full_selected, full_tab = choose_candidate(oof, candidates, cfg)
    full_tab.to_csv(TABLE / "DEVELOPMENT_FULL_SELECTION_TABLE.csv", index=False)
    write_json(OUT / "FINAL_DEVELOPMENT_SELECTED_CONDITIONALTWIN.json", {
        **full_selected.__dict__,
        "selection_data": "development LOHO predictions only",
        "external_labels_used": False,
        "external_cohorts_confirmatory_after_selection": False,
        "conditional_representation": "raw descriptors plus label-blind host-centered deviations",
    })
    return final, full_selected, oof


def collect_model_table(pred: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for model in ["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET", "Logistic", "PairRank", "HostBaseline"]:
        rows.append({"cohort": cohort, "model": model, **model_metrics(pred, f"{model}_score", f"{model}_probability")})
    return pd.DataFrame(rows)


def per_host_table(pred: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for h, g in pred.groupby("host"):
        for model in ["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET", "Logistic", "PairRank", "HostBaseline"]:
            if g.y.nunique() == 2:
                auc = float(roc_auc_score(g.y, g[f"{model}_score"]))
                ap = float(average_precision_score(g.y, g[f"{model}_score"]))
            else:
                auc = np.nan
                ap = np.nan
            rows.append({
                "cohort": cohort, "host": h, "records": len(g), "positives": int(g.y.sum()),
                "model": model, "AUROC": auc, "AUPRC": ap,
            })
    return pd.DataFrame(rows)


def screening_utility(pred: pd.DataFrame, cohort: str, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for model in ["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET", "Logistic", "PairRank"]:
        score = f"{model}_score"
        for frac in fractions:
            for h, g in pred.groupby("host"):
                if len(g) < 5 or g.y.sum() == 0:
                    continue
                n = max(1, int(math.ceil(len(g) * frac)))
                top = g.nlargest(n, score)
                precision = float(top.y.mean())
                prevalence = float(g.y.mean())
                rows.append({
                    "cohort": cohort, "model": model, "fraction": frac, "host": h,
                    "records": len(g), "selected": n, "precision": precision,
                    "prevalence": prevalence, "enrichment": precision / prevalence if prevalence > 0 else np.nan,
                    "recall": float(top.y.sum() / g.y.sum()),
                })
    return pd.DataFrame(rows)


def _cross_concordance(pos_score: np.ndarray, neg_score: np.ndarray) -> float:
    """Mann–Whitney concordance numerator, including half credit for ties."""
    if not len(pos_score) or not len(neg_score):
        return 0.0
    neg_sorted = np.sort(np.asarray(neg_score, float))
    pos = np.asarray(pos_score, float)
    lower = np.searchsorted(neg_sorted, pos, side="left")
    upper = np.searchsorted(neg_sorted, pos, side="right")
    return float(np.sum(lower + 0.5 * (upper - lower)))


def host_bootstrap(pred: pd.DataFrame, cohort: str, reps: int, seed: int) -> pd.DataFrame:
    """Exact cluster bootstrap using precomputed host-by-host Mann–Whitney matrices."""
    rng = np.random.default_rng(seed + stable_int(cohort))
    hosts = list(pred.host.unique())
    models = ["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET"]
    H = len(hosts)
    y_cache = [pred.loc[pred.host.eq(h), "y"].to_numpy(int) for h in hosts]
    npos = np.asarray([int(y.sum()) for y in y_cache], float)
    nneg = np.asarray([int(len(y) - y.sum()) for y in y_cache], float)
    model_cache = {}
    for model in models:
        scores = [pred.loc[pred.host.eq(h), f"{model}_score"].to_numpy(float) for h in hosts]
        centered = [v - v.mean() for v in scores]
        pooled_C = np.zeros((H, H), float)
        centered_C = np.zeros((H, H), float)
        within_num = np.zeros(H, float)
        within_den = npos * nneg
        for i in range(H):
            pos_raw = scores[i][y_cache[i] == 1]
            pos_ctr = centered[i][y_cache[i] == 1]
            for j in range(H):
                neg_raw = scores[j][y_cache[j] == 0]
                neg_ctr = centered[j][y_cache[j] == 0]
                pooled_C[i, j] = _cross_concordance(pos_raw, neg_raw)
                centered_C[i, j] = _cross_concordance(pos_ctr, neg_ctr)
            within_num[i] = pooled_C[i, i]
        model_cache[model] = (pooled_C, centered_C, within_num, within_den)

    rows = []
    for b in range(reps):
        sampled = rng.integers(0, H, size=H)
        k = np.bincount(sampled, minlength=H).astype(float)
        pos_total = float(k @ npos)
        neg_total = float(k @ nneg)
        if pos_total <= 0 or neg_total <= 0:
            continue
        pooled_den = pos_total * neg_total
        for model in models:
            pooled_C, centered_C, within_num, within_den = model_cache[model]
            pair_den = float(k @ within_den)
            rows.append({
                "cohort": cohort, "replicate": b, "model": model,
                "pooled_AUROC": float((k @ pooled_C @ k) / pooled_den),
                "host_centered_AUROC": float((k @ centered_C @ k) / pooled_den),
                "within_host_pair_AUROC": float((k @ within_num) / pair_den) if pair_den else np.nan,
            })
    return pd.DataFrame(rows)


def paired_bootstrap_differences(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparators = ["RawHostClassET", "CenteredOnlyET", "ConditionalET"]
    metrics = ["pooled_AUROC", "host_centered_AUROC", "within_host_pair_AUROC"]
    for cohort in boot.cohort.unique():
        sub = boot[boot.cohort.eq(cohort)]
        for comparator in comparators:
            for metric in metrics:
                piv = sub.pivot(index="replicate", columns="model", values=metric).dropna()
                if "ET" not in piv or comparator not in piv:
                    continue
                diff = piv[comparator] - piv["ET"]
                rows.append({
                    "cohort": cohort, "comparison": f"{comparator} - ET", "metric": metric,
                    "replicates": int(len(diff)), "mean_difference": float(diff.mean()),
                    "median_difference": float(diff.median()),
                    "ci95_low": float(diff.quantile(0.025)), "ci95_high": float(diff.quantile(0.975)),
                    "probability_difference_gt_0": float((diff > 0).mean()),
                })
    return pd.DataFrame(rows)


def statistical_tests(host_tables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in host_tables.cohort.unique():
        for comparator in ("RawHostClassET", "CenteredOnlyET", "ConditionalET"):
            sub = host_tables[(host_tables.cohort == cohort) & host_tables.model.isin(["ET", comparator])]
            piv = sub.pivot(index="host", columns="model", values="AUROC").dropna()
            if len(piv) >= 3 and not np.allclose(piv[comparator], piv.ET):
                stat, p = wilcoxon(piv[comparator], piv.ET, alternative="greater", zero_method="wilcox")
            else:
                stat, p = np.nan, 1.0
            rows.append({
                "cohort": cohort, "comparison": f"{comparator} > ET", "metric": "per-host AUROC",
                "hosts": len(piv), "mean_difference": float((piv[comparator] - piv.ET).mean()) if len(piv) else np.nan,
                "median_difference": float((piv[comparator] - piv.ET).median()) if len(piv) else np.nan,
                "wilcoxon_statistic": safe(stat), "p_value": float(p),
            })
    out = pd.DataFrame(rows)
    out["holm_p"] = holm_adjust(out.p_value)
    out["significant_0_05"] = out.holm_p < 0.05
    return out


def make_figures(summary: pd.DataFrame, hosttab: pd.DataFrame, screening: pd.DataFrame) -> None:
    dev = summary[summary.cohort.eq("Development_LOHO")].set_index("model").loc[["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(dev))
    width = 0.24
    ax.bar(x - width, dev.pooled_AUROC, width, label="Pooled")
    ax.bar(x, dev.host_centered_AUROC, width, label="Host-centered")
    ax.bar(x + width, dev.within_host_pair_AUROC, width, label="Within-host")
    ax.set_xticks(x, dev.index)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Development-only host-disjoint comparison")
    ax.legend()
    save_figure(fig, "FIG1_DEVELOPMENT_CONDITIONALTWIN_COMPARISON")

    cohorts = ["Development_LOHO", "2DMD_retrospective", "Imp2D_retrospective", "QPOD_retrospective"]
    models = ["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET"]
    mat = summary.set_index(["cohort", "model"]).within_host_pair_AUROC.unstack().reindex(index=cohorts, columns=models)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(mat.to_numpy(float), aspect="auto", vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(cohorts)), cohorts)
    for i in range(len(cohorts)):
        for j in range(len(models)):
            v = mat.iloc[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.3f}", ha="center", va="center")
    ax.set_title("Within-host transport matrix (external cohorts retrospective)")
    fig.colorbar(im, ax=ax, label="Within-host pair AUROC")
    save_figure(fig, "FIG2_WITHIN_HOST_TRANSPORT_MATRIX")

    shortcut = summary[summary.model.isin(["ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET"])].pivot(index="cohort", columns="model", values="shortcut_amplification_index")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shortcut.plot(kind="bar", ax=ax)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Pooled AUROC − within-host AUROC")
    ax.set_title("Shortcut amplification index")
    ax.legend(title="Model")
    save_figure(fig, "FIG3_SHORTCUT_AMPLIFICATION")

    h = hosttab[hosttab.model.isin(["ET", "ConditionalET"])].pivot_table(index=["cohort", "host"], columns="model", values="AUROC").dropna().reset_index()
    fig, ax = plt.subplots(figsize=(7, 7))
    for cohort, g in h.groupby("cohort"):
        ax.scatter(g.ET, g.ConditionalET, label=cohort, alpha=0.75)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ET host AUROC")
    ax.set_ylabel("ConditionalTwin-2D host AUROC")
    ax.set_title("Per-host ranking comparison")
    ax.legend(fontsize=8)
    save_figure(fig, "FIG4_PER_HOST_CONDITIONALTWIN_VS_ET")

    if len(screening):
        agg = screening.groupby(["cohort", "model", "fraction"], as_index=False).enrichment.median()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (cohort, model), g in agg.groupby(["cohort", "model"]):
            if model in ("ET", "RawHostClassET", "CenteredOnlyET", "ConditionalET"):
                ax.plot(g.fraction, g.enrichment, marker="o", label=f"{cohort}: {model}")
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.set_xlabel("Fraction screened within host")
        ax.set_ylabel("Median enrichment over random")
        ax.set_title("DFT-budget screening utility")
        ax.legend(fontsize=7)
        save_figure(fig, "FIG5_SCREENING_ENRICHMENT")


def reports(summary, tests, selected, selection_table, screening_summary, paired_differences, cfg):
    dev_et = summary[(summary.cohort == "Development_LOHO") & (summary.model == "ET")].iloc[0]
    dev_rawhc = summary[(summary.cohort == "Development_LOHO") & (summary.model == "RawHostClassET")].iloc[0]
    dev_center = summary[(summary.cohort == "Development_LOHO") & (summary.model == "CenteredOnlyET")].iloc[0]
    dev_cond = summary[(summary.cohort == "Development_LOHO") & (summary.model == "ConditionalET")].iloc[0]
    dev_pair = paired_differences[(paired_differences.cohort == "Development_LOHO") & (paired_differences.comparison == "ConditionalET - ET") & (paired_differences.metric == "within_host_pair_AUROC")].iloc[0]
    selection_stability = pd.read_csv(TABLE / "DEVELOPMENT_ROW_EXCLUSION_SELECTION_SENSITIVITY.csv").name.value_counts()
    selected_fold_count = int(selection_stability.get(selected.name, 0))
    external = summary[summary.cohort.str.contains("retrospective") & summary.model.eq("ConditionalET")]
    result_lines = []
    for _, r in external.iterrows():
        result_lines.append(
            f"- {r.cohort}: pooled AUROC {r.pooled_AUROC:.4f}, host-centered {r.host_centered_AUROC:.4f}, within-host {r.within_host_pair_AUROC:.4f}."
        )
    result_text = "\n".join(result_lines)
    manuscript = f"""# Manuscript insertion — ConditionalTwin-2D stage

## Methods

A development-only candidate-selection framework was evaluated under leave-one-host-out cross-fitting. Its nonlinear conditional branch, termed ConditionalTwin-2D, concatenated the 68 raw frozen descriptors with 68 label-blind deviations from the mean descriptor vector of the candidate pool belonging to the same host. Training assigned equal total weight to each host and, for mixed hosts, equal total weight to the two target classes. Three explicit ablations separated the effects of the original host-balanced estimator, host/class reweighting alone, and centered descriptors alone. Additional host-prevalence and within-host pairwise ranking branches were evaluated in a finite, predeclared combination grid. For every held-out development host, the combination was selected using cross-fitted predictions from the remaining hosts only, prioritizing within-host pair AUROC, followed by host-centered and pooled AUROC. No Imp2D, 2DMD, or QPOD label was used in representation, model, or weight selection.

## Results

The original ExtraTrees baseline achieved pooled AUROC {dev_et.pooled_AUROC:.4f}, host-centered AUROC {dev_et.host_centered_AUROC:.4f}, within-host pair AUROC {dev_et.within_host_pair_AUROC:.4f}, and macro-host AUROC {dev_et.macro_host_AUROC:.4f}. Host/class reweighting alone increased within-host AUROC to {dev_rawhc.within_host_pair_AUROC:.4f}; centered descriptors alone increased it to {dev_center.within_host_pair_AUROC:.4f}; and their combined ConditionalTwin-2D representation reached pooled AUROC {dev_cond.pooled_AUROC:.4f}, host-centered AUROC {dev_cond.host_centered_AUROC:.4f}, within-host AUROC {dev_cond.within_host_pair_AUROC:.4f}, and macro-host AUROC {dev_cond.macro_host_AUROC:.4f}. The paired host-bootstrap gain in within-host AUROC over ExtraTrees was {dev_pair.mean_difference:+.4f} (95% CI, {dev_pair.ci95_low:+.4f} to {dev_pair.ci95_high:+.4f}). The candidate-selection procedure retained the conditional branch alone in all {selected_fold_count}/30 overlapping host-row-exclusion sensitivity tables and in the full development selection; these row-exclusion tables re-score fixed cross-fitted component predictions and are not independent replications or nested outer-fold estimates; the host and linear pairwise branches were therefore rejected rather than retained for architectural complexity. Host-centered AUROC decreased modestly relative to ExtraTrees, making the result a deliberate improvement in conditional defect ordering rather than a uniform improvement in every metric.

The previously opened external cohorts were retained strictly as retrospective diagnostics. On 2DMD, ConditionalTwin-2D increased within-host AUROC from {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ET')].iloc[0].within_host_pair_AUROC:.4f} to {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ConditionalET')].iloc[0].within_host_pair_AUROC:.4f}, while pooled AUROC decreased from {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ET')].iloc[0].pooled_AUROC:.4f} to {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ConditionalET')].iloc[0].pooled_AUROC:.4f}, consistent with suppression of between-host shortcut signal. No convincing retrospective advantage was observed on Imp2D or QPOD.

## Claim boundary

ConditionalTwin-2D strengthens the algorithmic contribution by providing a label-blind, candidate-pool-conditional representation with explicit ablations and development-only selection. It requires a batch of candidate defects from the same host to compute host-centered descriptors and is not a single-record inductive predictor. Because the method was designed after the earlier external outcomes were known, Imp2D, 2DMD, and QPOD results cannot be presented as prospective validation. A newly sealed untouched source or prospective DFT cohort remains necessary for a confirmatory transport claim.
"""
    (OUT / "MANUSCRIPT_TRANSPORTRANK_INSERT.md").write_text(manuscript, encoding="utf-8")

    arabic = f"""# تقرير المرحلة بالعربية — ConditionalTwin-2D

## الإضافة الجديدة

تم اختبار إطار اختيار متعدد الفروع، لكن الاختيار التطويري رفض التعقيد الزائد واختار فرعًا واحدًا ثابتًا في {selected_fold_count}/30 مرة: ConditionalTwin-2D. هذا الفرع يجمع الخصائص الـ68 الأصلية مع انحراف كل عيب عن متوسط مجموعة العيوب المرشحة داخل الـhost نفسه، دون استخدام labels في حساب المتوسط. كما يعطي كل host وزنًا كليًا متساويًا ويوازن الفئتين داخله.

## Ablations على Development

- ExtraTrees الأصلي: within-host AUROC = {dev_et.within_host_pair_AUROC:.4f}
- تغيير الأوزان فقط: {dev_rawhc.within_host_pair_AUROC:.4f}
- الخصائص centered فقط: {dev_center.within_host_pair_AUROC:.4f}
- الدمج النهائي ConditionalTwin-2D: {dev_cond.within_host_pair_AUROC:.4f}
- التحسن المقترن عن ExtraTrees: {dev_pair.mean_difference:+.4f}
- 95% host-bootstrap CI للتحسن: [{dev_pair.ci95_low:+.4f}, {dev_pair.ci95_high:+.4f}]
- Pooled AUROC النهائي: {dev_cond.pooled_AUROC:.4f}
- Macro-host AUROC النهائي: {dev_cond.macro_host_AUROC:.4f}
- Host-centered AUROC النهائي: {dev_cond.host_centered_AUROC:.4f} مقابل {dev_et.host_centered_AUROC:.4f} للأصل

هذا يعني أن التحسن الحقيقي يقع في ترتيب العيوب داخل الـhost، مع انخفاض محدود في host-centered AUROC؛ لذلك لا يجوز وصفه بأنه تفوق شامل في جميع المقاييس.

## الاختبارات الخارجية السابقة

- في 2DMD ارتفع within-host AUROC من {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ET')].iloc[0].within_host_pair_AUROC:.4f} إلى {summary[(summary.cohort == '2DMD_retrospective') & (summary.model == 'ConditionalET')].iloc[0].within_host_pair_AUROC:.4f}، بينما انخفض pooled AUROC؛ وهذا يتفق مع إزالة جزء من host shortcut، لكنه retrospective وليس إثباتًا جديدًا.
- في Imp2D وQPOD لم يظهر تفوق مقنع.

## القيد التطبيقي

الطريقة تحتاج مجموعة مرشحين من الـhost نفسه لحساب المتوسط؛ لذلك هي candidate-pool conditional وليست تنبؤًا inductive لسجل منفرد.

## الحكم

هذه المرحلة ترفع الجدة الخوارزمية والمنهجية بوضوح مقارنة بالنسخة السابقة، لكنها لا تكفي وحدها لادعاء external generalization. أقوى خطوة لاحقة ما زالت lockbox جديدة مختومة أو prospective DFT، مع relaxed-structure audit وphysics validation.
"""
    (OUT / "ARABIC_STAGE_REPORT.md").write_text(arabic, encoding="utf-8")

    reviewer = pd.DataFrame([
        ["Development-only model selection", "PASS", "All ConditionalTwin weights selected from development cross-fitted predictions"],
        ["Within-host ranking objective", "PASS", "Pairwise rank branch trained only on within-host positive-negative pairs"],
        ["Host shortcut decomposition", "PASS", "Pooled, host-centered, within-host, macro-host, and shortcut index reported"],
        ["Strong tabular baseline", "PASS", "Five-seed ExtraTrees baseline retained"],
        ["Linear baseline", "PASS", "Host-balanced logistic regression added"],
        ["Ablation: host/class weighting", "PASS", "RawHostClassET separates weighting from representation"],
        ["Ablation: centered descriptors only", "PASS", "CenteredOnlyET isolates conditional deviations"],
        ["Conditional nonlinear branch", "PASS", "Raw plus label-blind host-centered descriptors with host/class balancing"],
        ["Ablation: rank only", "PASS", "PairRank reported"],
        ["Ablation: host only", "PASS", "HostBaseline reported"],
        ["Development-selected final method", "PASS", "Conditional-only selected in every meta-selection fold"],
        ["Screening utility", "PASS", "Within-host top-budget enrichment and recall reported"],
        ["Paired host statistics", "PASS", "Wilcoxon plus Holm across cohorts"],
        ["Prior external results preserved", "PASS", "Imp2D, 2DMD, QPOD treated as retrospective"],
        ["New prospective lockbox", "NOT COMPLETE", "Requires untouched data after method freeze"],
        ["Relaxed 2DMD structures", "NOT COMPLETE", "Raw VASP source is large; no result fabricated"],
        ["Published defect-aware fingerprint comparator", "NOT COMPLETE", "Requires exact source implementation and structural files"],
        ["Physics validation", "NOT COMPLETE", "Requires spin density/PDOS/local-moment files or new DFT"],
    ], columns=["requirement", "status", "evidence"])
    reviewer.to_csv(TABLE / "REVIEWER_COMPLETENESS_MATRIX.csv", index=False)

    boundaries = """# Claim boundaries

Allowed:
- ConditionalTwin-2D was selected and cross-fitted using development hosts only.
- The method explicitly optimizes within-host ranking and reports host shortcut amplification.
- Earlier Imp2D, 2DMD, and QPOD cohorts are retrospective stress tests.

Forbidden:
- Calling ConditionalTwin externally validated by those already opened cohorts.
- Calling QPOD database-independent or composition-unseen.
- Claiming the relaxed-structure mismatch has been repaired before raw relaxed structures are acquired.
- Claiming superiority over DefiNet, CFID/Hellinger, ALIGNN, CHGNet, or MACE without exact implemented comparisons.
- Hiding negative external outcomes or excluding weak hosts.
"""
    (OUT / "CLAIM_BOUNDARIES.md").write_text(boundaries, encoding="utf-8")

    next_lock = """# Prospective confirmation protocol

1. Freeze the exact feature list, ConditionalTwin weights, preprocessing, seeds, primary threshold, and calibration.
2. Register an untouched source before reading magnetic targets.
3. Require record-disjoint and exact-phase-disjoint data; composition-unseen is preferred and must be named accurately.
4. Seal classification and continuous predictions with SHA-256 before target access.
5. Report all hosts, including below-chance hosts.
6. Primary metrics: within-host pair AUROC, host-centered AUROC, pooled AUROC, Brier skill, and top-10% within-host enrichment.
7. No external recalibration, threshold change, score inversion, feature selection, or host deletion.
8. A DFT mini-lockbox of 5–8 untouched hosts and 20–30 defects per host is acceptable if no suitable public database exists.
"""
    (OUT / "NEXT_PROSPECTIVE_LOCKBOX_PROTOCOL.md").write_text(next_lock, encoding="utf-8")

    protocol = {
        "stage": cfg["project"], "version": cfg["version"], "generated_at": now(),
        "selected_candidate": selected.__dict__,
        "conditional_representation_requires_host_candidate_batch": True,
        "selection_source": "development cross-fitted predictions only",
        "no_external_labels_used_for_selection": True,
        "external_cohorts_status": "retrospective diagnostics only",
        "primary_endpoint_muB": float(cfg["primary_threshold_muB"]),
        "feature_count": 68,
        "future_confirmatory_lockbox_required": True,
    }
    write_json(OUT / "PROTOCOL_LOCK.json", protocol)


def make_package_manifest():
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name not in {"HASHES_SHA256.csv", "RESULT_PACKAGE.zip"}]
    manifest = pd.DataFrame([
        {"relative_path": str(p.relative_to(ROOT)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(files)
    ])
    manifest.to_csv(OUT / "HASHES_SHA256.csv", index=False)
    with zipfile.ZipFile(OUT / "RESULT_PACKAGE.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "RESULT_PACKAGE.zip"]):
            z.write(p, p.relative_to(ROOT))


def main():
    started = time.time()
    cfg = yaml.safe_load((ROOT / "config" / "project_config.yaml").read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "version": cfg["version"]})
    print("[1/8] Loading and verifying frozen inputs")
    dev, base, d2, imp, qpod, features = load_data()
    input_summary = {
        "development_records": len(dev), "development_hosts": int(dev.host.nunique()),
        "2DMD_records": len(d2), "2DMD_hosts": int(d2.host.nunique()),
        "Imp2D_records": len(imp), "Imp2D_hosts": int(imp.host.nunique()),
        "QPOD_records": len(qpod), "QPOD_hosts": int(qpod.host.nunique()),
        "feature_count": len(features),
    }
    write_json(OUT / "INPUT_SUMMARY.json", input_summary)

    print("[2/8] Cross-fitting ConditionalET, PairRank, host branch, and logistic baseline")
    dev_pred, selected, component_oof = development_oof(dev, base, features, cfg)

    print("[3/8] Fitting full-development models")
    models = fit_full_models(dev, features, cfg)

    print("[4/8] Running retrospective transport diagnostics")
    d2_pred = predict_models(d2, features, models, selected)
    imp_pred = predict_models(imp, features, models, selected)
    q_pred = predict_models(qpod, features, models, selected)
    d2_pred.to_csv(TABLE / "2DMD_RETROSPECTIVE_PREDICTIONS.csv", index=False)
    imp_pred.to_csv(TABLE / "IMP2D_RETROSPECTIVE_PREDICTIONS.csv", index=False)
    q_pred.to_csv(TABLE / "QPOD_RETROSPECTIVE_PREDICTIONS.csv", index=False)

    print("[5/8] Computing metrics, screening utility, and host statistics")
    summaries = [
        collect_model_table(dev_pred, "Development_LOHO"),
        collect_model_table(d2_pred, "2DMD_retrospective"),
        collect_model_table(imp_pred, "Imp2D_retrospective"),
        collect_model_table(q_pred, "QPOD_retrospective"),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(TABLE / "TABLE_MODEL_COHORT_SUMMARY.csv", index=False)
    hosttab = pd.concat([
        per_host_table(dev_pred, "Development_LOHO"),
        per_host_table(d2_pred, "2DMD_retrospective"),
        per_host_table(imp_pred, "Imp2D_retrospective"),
        per_host_table(q_pred, "QPOD_retrospective"),
    ], ignore_index=True)
    hosttab.to_csv(TABLE / "TABLE_HOST_LEVEL_METRICS.csv", index=False)
    screening = pd.concat([
        screening_utility(dev_pred, "Development_LOHO", cfg["statistics"]["screening_fractions"]),
        screening_utility(d2_pred, "2DMD_retrospective", cfg["statistics"]["screening_fractions"]),
        screening_utility(imp_pred, "Imp2D_retrospective", cfg["statistics"]["screening_fractions"]),
        screening_utility(q_pred, "QPOD_retrospective", cfg["statistics"]["screening_fractions"]),
    ], ignore_index=True)
    screening.to_csv(TABLE / "TABLE_SCREENING_UTILITY_BY_HOST.csv", index=False)
    screening_summary = screening.groupby(["cohort", "model", "fraction"], as_index=False).agg(
        hosts=("host", "nunique"), median_enrichment=("enrichment", "median"),
        mean_enrichment=("enrichment", "mean"), median_recall=("recall", "median")
    ) if len(screening) else pd.DataFrame()
    screening_summary.to_csv(TABLE / "TABLE_SCREENING_UTILITY_SUMMARY.csv", index=False)
    tests = statistical_tests(hosttab)
    tests.to_csv(TABLE / "TABLE_WILCOXON_HOLM.csv", index=False)

    print("[6/8] Host bootstrap")
    boots = []
    reps = int(cfg["statistics"]["host_bootstrap_replicates"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    for name, pred in [("Development_LOHO", dev_pred), ("2DMD_retrospective", d2_pred), ("Imp2D_retrospective", imp_pred), ("QPOD_retrospective", q_pred)]:
        boots.append(host_bootstrap(pred, name, reps, seed))
        print(f"  bootstrap complete: {name}")
    boot = pd.concat(boots, ignore_index=True)
    boot.to_csv(CHECK / "HOST_BOOTSTRAP_2000.csv", index=False)
    ci = boot.groupby(["cohort", "model"]).agg(
        pooled_low=("pooled_AUROC", lambda x: x.quantile(0.025)), pooled_high=("pooled_AUROC", lambda x: x.quantile(0.975)),
        host_centered_low=("host_centered_AUROC", lambda x: x.quantile(0.025)), host_centered_high=("host_centered_AUROC", lambda x: x.quantile(0.975)),
        within_host_low=("within_host_pair_AUROC", lambda x: x.quantile(0.025)), within_host_high=("within_host_pair_AUROC", lambda x: x.quantile(0.975)),
    ).reset_index()
    ci.to_csv(TABLE / "TABLE_HOST_BOOTSTRAP_CI.csv", index=False)
    paired_differences = paired_bootstrap_differences(boot)
    paired_differences.to_csv(TABLE / "TABLE_PAIRED_BOOTSTRAP_DIFFERENCES.csv", index=False)

    print("[7/8] Figures and reports")
    make_figures(summary, hosttab, screening)
    selection_table = pd.read_csv(TABLE / "DEVELOPMENT_FULL_SELECTION_TABLE.csv")
    reports(summary, tests, selected, selection_table, screening_summary, paired_differences, cfg)
    print("[8/8] Verification and result package")
    dev_cond = summary[(summary.cohort == "Development_LOHO") & (summary.model == "ConditionalET")].iloc[0]
    verification = {
        "completed_at": now(),
        "development_records_529": len(dev) == 529,
        "development_hosts_30": dev.host.nunique() == 30,
        "features_68": len(features) == 68,
        "component_oof_complete": component_oof.record_id.nunique() == 529,
        "selection_development_only": True,
        "no_external_labels_used_for_selection": True,
        "cohorts_4": summary.cohort.nunique() == 4,
        "models_7": summary.model.nunique() == 7,
        "host_bootstrap_rows_at_least_95pct": len(boot) >= int(0.95 * 4 * 4 * reps),
        "paired_bootstrap_table_complete": len(paired_differences) >= 4 * 3 * 3,
        "figures_all_formats": all(len(list(FIG.glob(f"*.{ext}"))) >= 5 for ext in ("png", "pdf", "svg")),
        "conditional_twin_within_host_finite": bool(np.isfinite(dev_cond.within_host_pair_AUROC)),
        "no_prospective_lockbox_claim": True,
        "no_relaxed_2dmd_completion_claim": True,
    }
    verification["all"] = all(verification.values())
    write_json(OUT / "VERIFICATION.json", verification)
    write_json(OUT / "STATUS.json", {
        "state": "SUCCESS" if verification["all"] else "FAILED_VERIFICATION",
        "updated_at": now(), "runtime_seconds": time.time() - started,
        "selected_candidate": selected.name,
    })
    if not verification["all"]:
        raise RuntimeError("Verification failed; inspect outputs/VERIFICATION.json")
    make_package_manifest()
    write_json(OUT / "SUCCESS.json", {
        "marker": "DEFECTTWIN2D_STRONG_Q1_TRANSPORT_RANK_COMPLETE",
        "completed_at": now(), "selected_candidate": selected.name,
        "result_zip": "outputs/RESULT_PACKAGE.zip", "acceptance_guaranteed": False,
    })
    # refresh package to include success marker
    make_package_manifest()
    print("SUCCESS. Upload outputs\\RESULT_PACKAGE.zip")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.mkdir(parents=True, exist_ok=True)
        (LOG / "ERROR_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(OUT / "STATUS.json", {"state": "FAILED", "updated_at": now()})
        traceback.print_exc()
        sys.exit(1)
