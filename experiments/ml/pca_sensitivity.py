#!/usr/bin/env python
"""
PCA-sensitivity experiment.

Usage:
    python experiments/ml/pca_sensitivity.py -c experiments/configs/pca_sensitivity_newlabels.yaml
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

from src.harmonization import make_harmonizer
from src.utils.cv_metrics import get_scores_binary, get_scores_multiclass
from src.utils.data_prep import (
    append_results_csv,
    load_experiment_data,
    prepare_pathology_labels,
)


SITE_COLUMN = "hospital_id"
COVARIATES = ["age", "gender"]

logger = logging.getLogger("pca_sensitivity")


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p.add_argument("--task", choices=["site", "patho", "both"], default="both",
                   help="Restrict to one task (default: both)")
    p.add_argument("--models", type=_csv_list, default=None,
                   help="Comma-separated model subset, e.g. 'catboost' or 'logreg'. "
                        "Default: all in config.")
    p.add_argument("--methods", type=_csv_list, default=None,
                   help="Comma-separated method subset, e.g. 'raw,sitewise,combat'. "
                        "Default: all in config.")
    p.add_argument("--pca-vars", type=_csv_list, default=None, dest="pca_vars",
                   help="Comma-separated pca_var subset, e.g. 'none,0.95,0.80'. "
                        "Default: all in config.")
    p.add_argument("--results-suffix", default="",
                   help="Optional suffix for output CSV filenames (e.g. '_wigner') "
                        "to avoid collisions when multiple machines write concurrently.")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_scaling_and_pca(X_train, X_test, pca_variance, X_calib=None, random_state=42):
    """Robust scaler + PCA fit on train (with optional calibration concatenated for harmonizer fit later)."""
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_calib_s = scaler.transform(X_calib) if X_calib is not None else None

    if pca_variance == "none":
        X_train_p, X_test_p, X_calib_p = X_train_s, X_test_s, X_calib_s
        n_comps = X_train.shape[1]
    else:
        n_components = None if pca_variance == "all" else float(pca_variance)
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        X_calib_p = pca.transform(X_calib_s) if X_calib_s is not None else None
        n_comps = pca.n_components_

    X_train_p = pd.DataFrame(X_train_p, index=X_train.index)
    X_test_p = pd.DataFrame(X_test_p, index=X_test.index)
    X_calib_p = pd.DataFrame(X_calib_p, index=X_calib.index) if X_calib is not None else None
    return X_train_p, X_test_p, X_calib_p, n_comps


def build_classifier(model_name, catboost_params, random_state):
    if model_name == "catboost":
        return CatBoostClassifier(**catboost_params)
    if model_name == "logreg":
        return LogisticRegression(max_iter=20000, random_state=random_state, C=1.0)
    if model_name == "svm":
        return SVC(kernel="rbf", probability=True, C=1.0, random_state=random_state)
    raise ValueError(f"Unknown model: {model_name}")


def run_site_classification(X, y, info_df, method, pca_var, model_name, cfg):
    """5-fold stratified CV for site classification."""
    skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits_site"], shuffle=True,
                          random_state=cfg["cv"]["random_state"])
    sites = info_df[SITE_COLUMN]
    cov = info_df[COVARIATES]
    le = LabelEncoder().fit(sites.unique())

    detailed = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_pca, X_test_pca, _, n_features = apply_scaling_and_pca(
            X_train, X_test, pca_var, random_state=cfg["cv"]["random_state"]
        )

        harmonizer = make_harmonizer(method, sites, cov_df=cov)
        if harmonizer:
            harmonizer.fit(X_train_pca)
            X_train_harm = harmonizer.transform(X_train_pca)
            X_test_harm = harmonizer.transform(X_test_pca)
        else:
            X_train_harm, X_test_harm = X_train_pca, X_test_pca

        catboost_params = cfg["catboost_params_site"] if model_name == "catboost" else {}
        clf = build_classifier(model_name, catboost_params, cfg["cv"]["random_state"])
        if model_name == "catboost":
            clf.fit(X_train_harm, y_train, verbose=False)
        else:
            clf.fit(X_train_harm, y_train)

        y_prob = clf.predict_proba(X_test_harm)
        scores_fold = get_scores_multiclass(y_test, y_prob, le)
        scores_fold.update({
            "model": model_name, "method": method, "fold_id": fold,
            "pca_var": pca_var, "n_features": n_features,
        })
        detailed.append(scores_fold)
    return detailed


def run_pathology_classification(X, y, info_df, method, pca_var, model_name, cfg):
    """LOSO CV with k-sample harmonizer calibration."""
    groups = info_df[SITE_COLUMN]
    logo = LeaveOneGroupOut()
    detailed = []
    k_calib = cfg["cv"]["k_calibration"]
    rs = cfg["cv"]["random_state"]

    for fold, (train_idx, site_idx) in enumerate(logo.split(X, y, groups)):
        hospital_test = groups.iloc[site_idx].unique()[0]
        X_train_pool = X.iloc[train_idx]
        y_train_pool = y.iloc[train_idx]
        X_site = X.iloc[site_idx]
        y_site = y.iloc[site_idx]

        site_norm_mask = (y_site == 0)
        X_site_norm = X_site[site_norm_mask]
        y_site_norm = y_site[site_norm_mask]

        if len(X_site_norm) < k_calib:
            logger.warning(f"Hospital {hospital_test}: only {len(X_site_norm)} normals (< {k_calib}); using all for calib")
            X_calib = X_site_norm
            X_test_norm = pd.DataFrame(columns=X.columns)
            y_test_norm = pd.Series(dtype=int)
        else:
            calib_idx, test_idx = train_test_split(X_site_norm.index, train_size=k_calib, random_state=rs)
            X_calib, X_test_norm = X_site_norm.loc[calib_idx], X_site_norm.loc[test_idx]
            y_test_norm = y_site_norm.loc[test_idx]

        X_test_full = pd.concat([X_test_norm, X_site[~site_norm_mask]])
        y_test_full = pd.concat([y_test_norm, y_site[~site_norm_mask]])

        # Leakage invariant
        calib_idx_set = set(X_calib.index)
        assert not calib_idx_set & set(X_train_pool.index), "calib leaked into train pool"
        assert not calib_idx_set & set(X_test_full.index), "calib leaked into test set"

        X_train_pca, X_test_pca, X_calib_pca, n_features = apply_scaling_and_pca(
            X_train_pool, X_test_full, pca_var, X_calib=X_calib, random_state=rs
        )

        train_norm_mask = (y_train_pool == 0)
        X_fit_harm = pd.concat([X_train_pca[train_norm_mask], X_calib_pca], axis=0)

        harmonizer = make_harmonizer(method, info_df[SITE_COLUMN], cov_df=info_df)
        if harmonizer:
            harmonizer.fit(X_fit_harm)
            X_train_harm = harmonizer.transform(X_train_pca)
            X_test_harm = harmonizer.transform(X_test_pca)
        else:
            X_train_harm, X_test_harm = X_train_pca, X_test_pca

        catboost_params = cfg["catboost_params_patho"] if model_name == "catboost" else {}
        clf = build_classifier(model_name, catboost_params, rs)
        if model_name == "catboost":
            clf.fit(X_train_harm, y_train_pool, verbose=False)
        else:
            clf.fit(X_train_harm, y_train_pool)
        y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]

        s = get_scores_binary(y_test_full, y_pred_proba)
        detailed.append({
            "mcc": s["MCC"], "accuracy": s["Accuracy"], "precision": s["Precision"],
            "recall": s["Recall"], "f1-score": s["F1-Score"], "auc": s["AUC"],
            "model": model_name, "hospital": hospital_test, "method": method,
            "n_calib": len(X_calib), "n_test": len(X_test_full),
            "pca_var": pca_var, "n_features": n_features,
        })
    return detailed


def _run_job(task, X, y, info_df, method, pca_var, model_name, cfg):
    logger.info(f"[START] {model_name} | {task} | {method} | pca={pca_var}")
    fn = run_site_classification if task == "site" else run_pathology_classification
    results = fn(X, y, info_df, method, pca_var, model_name, cfg)
    logger.info(f"[DONE]  {model_name} | {task} | {method} | pca={pca_var}")
    return task, method, pca_var, model_name, results


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "pca_sensitivity.log"
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["log_dir"])

    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {cfg.get('experiment_name', '<unnamed>')}")

    info, feats = load_experiment_data(cfg["paths"]["info_file"], cfg["paths"]["features_file"])
    y_patho = prepare_pathology_labels(info)
    y_site = info[SITE_COLUMN]
    logger.info(f"Loaded data: features {feats.shape}, info {len(info)} rows")

    norm_mask = (y_patho == 0)
    X_site_only = feats[norm_mask].reset_index(drop=True)
    y_site_only = y_site[norm_mask].reset_index(drop=True)
    info_site_only = info[norm_mask].reset_index(drop=True)

    methods = cfg["harmonization_methods"]
    pca_variants = cfg["pca_variants"]
    models = cfg["models"]

    # CLI overrides
    if args.methods:
        bad = set(args.methods) - set(methods)
        if bad:
            raise ValueError(f"--methods contains values not in config: {bad}")
        methods = args.methods
    if args.pca_vars:
        # Coerce numeric strings back to floats so they match config equality
        pca_variants_str = [str(v) for v in pca_variants]
        bad = set(args.pca_vars) - set(pca_variants_str)
        if bad:
            raise ValueError(f"--pca-vars contains values not in config: {bad}")
        pca_variants = [v for v in pca_variants if str(v) in args.pca_vars]
    if args.models:
        bad = set(args.models) - set(models)
        if bad:
            raise ValueError(f"--models contains values not in config: {bad}")
        models = args.models

    logger.info(f"Slice: models={models} methods={methods} pca_vars={pca_variants} task={args.task}")

    tasks = []
    if args.task in ("site", "both"):
        tasks.append("site")
    if args.task in ("patho", "both"):
        tasks.append("patho")

    job_args = []
    for method in methods:
        for pca_var in pca_variants:
            if "site" in tasks:
                job_args.append(("site", X_site_only, y_site_only, info_site_only, method, pca_var))
            if "patho" in tasks:
                job_args.append(("patho", feats, y_patho, info, method, pca_var))

    n_parallel = cfg.get("n_parallel", 1)
    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info(f"=== model: {model_name} ({len(job_args)} jobs, n_parallel={n_parallel}) ===")
        if n_parallel > 1 and model_name not in ("catboost",):
            jobs = [delayed(_run_job)(*a, model_name, cfg) for a in job_args]
            results = Parallel(n_jobs=n_parallel, verbose=10)(jobs)
        else:
            results = [_run_job(*a, model_name, cfg) for a in job_args]

        site_results, patho_results = [], []
        for task, _, _, _, rl in results:
            (site_results if task == "site" else patho_results).extend(rl)

        suffix = args.results_suffix
        if site_results:
            path = out_dir / f"pca_sensitivity_results_site_{model_name}{suffix}.csv"
            append_results_csv(pd.DataFrame(site_results), str(path))
            logger.info(f"  wrote {path}")
        if patho_results:
            path = out_dir / f"pca_sensitivity_results_patho_{model_name}{suffix}.csv"
            append_results_csv(pd.DataFrame(patho_results), str(path))
            logger.info(f"  wrote {path}")

    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
