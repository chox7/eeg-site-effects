#!/usr/bin/env python
"""
Feature-group ablation for site + patho classification on new-label data.

Conditions (raw features only — no harmonisation, since the paper's ablation
characterises the unharmonised site/patho signal carried by each group):
  - full_{coh,pow,cov}         : all features in that group (deterministic, R=1)
  - no_{coh,pow,cov}           : drop one group, train on the rest (R=1)
  - sub_{coh,pow,cov}          : random subsample of N=min_group_size features
                                  from that group; R repeats (R=1 for `cov`
                                  since cov has exactly N features, so the
                                  draw is deterministic).

Tasks:
  - site  : 5-fold stratified CV on ELM_n, one-vs-rest per-hospital MCC.
  - patho : 30-fold LOSO on ELM_p, raw features (no calibration since no
            harmoniser is fit).

Models: catboost (GPU) or logreg.

Output: appends one row per (condition, seed, fold/hospital) to
  results/tables/02_site_effect/feature_ablation_{task}_{model}.csv
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Add repo root to path for src.harmonization etc., though we don't use it here
REPO = Path(__file__).resolve().parents[1]

SITE_INFO = REPO / "data/ELM19/filtered/ELM19_info_filtered_norm_newlabels.csv"
SITE_FEAT = REPO / "data/ELM19/filtered/ELM19_features_filtered_norm_newlabels.csv"
PATHO_INFO = REPO / "data/ELM19/filtered/ELM19_info_filtered_newlabels.csv"
PATHO_FEAT = REPO / "data/ELM19/filtered/ELM19_features_filtered_newlabels.csv"

OUT_DIR = REPO / "results/tables/02_site_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATBOOST_PARAMS_SITE = dict(
    iterations=1500, learning_rate=0.07, depth=5, l2_leaf_reg=9,
    task_type="GPU", verbose=False, allow_writing_files=False,
    loss_function="MultiClass", eval_metric="MCC", random_seed=42)

CATBOOST_PARAMS_PATHO = dict(
    iterations=1500, learning_rate=0.07, depth=5, l2_leaf_reg=9,
    task_type="GPU", verbose=False, allow_writing_files=False,
    loss_function="Logloss", eval_metric="AUC", random_seed=42)

ALL_CONDITIONS = [
    "full_coh", "full_pow", "full_cov",
    "no_coh", "no_pow", "no_cov",
    "sub_coh", "sub_pow", "sub_cov",
]


def setup_logger(name="feature_ablation"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


def discover_groups(columns: list[str]) -> dict[str, list[str]]:
    return {
        "coh": [c for c in columns if c.startswith("coh_")],
        "pow": [c for c in columns if c.startswith("pow_")],
        "cov": [c for c in columns if c.startswith("cov_")],
    }


def select_columns(groups: dict[str, list[str]], condition: str,
                   n_features: int, seed: int) -> list[str]:
    if condition.startswith("full_"):
        return list(groups[condition.split("_", 1)[1]])
    if condition.startswith("no_"):
        drop = condition.split("_", 1)[1]
        return [c for g, cols in groups.items() if g != drop for c in cols]
    if condition.startswith("sub_"):
        grp = condition.split("_", 1)[1]
        cols = groups[grp]
        if len(cols) <= n_features:
            return list(cols)
        rng = np.random.default_rng(seed)
        return list(rng.choice(cols, size=n_features, replace=False))
    raise ValueError(f"unknown condition {condition!r}")


def build_model(model_name: str, task: str, random_state: int = 42):
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        params = dict(CATBOOST_PARAMS_SITE if task == "site" else CATBOOST_PARAMS_PATHO)
        params["random_seed"] = random_state
        return CatBoostClassifier(**params)
    if model_name == "logreg":
        return Pipeline([
            ("scale", RobustScaler()),
            ("clf", LogisticRegression(max_iter=20000, random_state=random_state, C=1.0)),
        ])
    raise ValueError(f"unknown model {model_name!r}")


# ---------------------- task implementations ------------------------------

def run_site_cv(X, y, cols, model_name, all_hospitals, logger):
    """One full 5-fold stratified CV. Returns list of per-fold dicts."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = build_model(model_name, "site")
        model.fit(X.iloc[tr][cols], y.iloc[tr])
        pred = model.predict(X.iloc[te][cols])
        if hasattr(pred, "flatten"):
            pred = np.asarray(pred).flatten()
        y_true = y.iloc[te].values
        overall = matthews_corrcoef(y_true, pred)
        row = {"fold": fold, "mcc_overall": float(overall),
               "fit_secs": round(time.time() - t0, 2)}
        for h in all_hospitals:
            y_b = (y_true == h).astype(int)
            p_b = (np.asarray(pred) == h).astype(int)
            row[f"mcc_{h}"] = float(matthews_corrcoef(y_b, p_b)) if y_b.sum() else float("nan")
        rows.append(row)
        logger.info(f"    fold {fold+1}/5  overall MCC={overall:.4f}  ({row['fit_secs']:.1f}s)")
    return rows


def run_patho_loso(X, y_patho, hospitals, cols, model_name, logger):
    """LOSO over 30 hospitals. Raw features, no harmoniser, no calibration."""
    rows = []
    hosp_list = sorted(hospitals.unique(), key=lambda h: int(str(h)[1:]))
    for i, h in enumerate(hosp_list, 1):
        t0 = time.time()
        tr = (hospitals != h).values
        te = (hospitals == h).values
        if y_patho[te].nunique() < 2 or y_patho[tr].nunique() < 2:
            logger.info(f"    LOSO {i:02d}/30 {h}: skipped (single-class)")
            continue
        model = build_model(model_name, "patho")
        model.fit(X.iloc[tr][cols], y_patho[tr])
        try:
            prob = model.predict_proba(X.iloc[te][cols])[:, 1]
        except AttributeError:
            prob = model.decision_function(X.iloc[te][cols])
        pred = (prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_patho[te], prob))
        except ValueError:
            auc = float("nan")
        mcc = float(matthews_corrcoef(y_patho[te], pred))
        secs = round(time.time() - t0, 2)
        rows.append({"hospital": h, "auc": auc, "mcc": mcc, "fit_secs": secs})
        logger.info(f"    LOSO {i:02d}/30 {h}: AUC={auc:.4f} MCC={mcc:.4f}  ({secs:.1f}s)")
    return rows


# ---------------------- driver --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["site", "patho"], required=True)
    parser.add_argument("--model", choices=["catboost", "logreg"], required=True)
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                        help="comma-separated conditions to run (default: all 9)")
    parser.add_argument("--n-features", type=int, default=190,
                        help="subsample size for sub_* conditions (default: 190 = cov group size)")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="random subsample repeats for sub_coh and sub_pow "
                             "(sub_cov is deterministic since N=group_size; default: 10)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="base seed for sub_* RNG; per-draw seed = seed_base + r (default: 42)")
    args = parser.parse_args()

    log = setup_logger()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            log.error(f"unknown condition: {c}; allowed = {ALL_CONDITIONS}")
            sys.exit(1)

    log.info(f"Task={args.task}  Model={args.model}")
    log.info(f"Conditions={conditions}")
    log.info(f"N_features={args.n_features}  R={args.n_repeats}  seed_base={args.seed_base}")

    # ---- load data once ----
    if args.task == "site":
        info_path, feat_path = SITE_INFO, SITE_FEAT
    else:
        info_path, feat_path = PATHO_INFO, PATHO_FEAT
    log.info(f"Loading info: {info_path.relative_to(REPO)}")
    info = pd.read_csv(info_path)
    log.info(f"Loading features: {feat_path.relative_to(REPO)}")
    feats = pd.read_csv(feat_path)
    log.info(f"  info rows={len(info)}  feats={feats.shape}")

    info = info.rename(columns={
        "age_dec": "age", "patient_sex": "gender",
        "institution_id": "hospital_id",
        "classification": "pathology_label",
    })

    groups = discover_groups(feats.columns.tolist())
    log.info(f"  feature groups: {{ {', '.join(f'{g}={len(c)}' for g, c in groups.items())} }}")

    # task-specific labels
    if args.task == "site":
        y = info["hospital_id"]
        all_hospitals = sorted(y.unique(), key=lambda h: int(str(h)[1:]))
        hospitals = None
        y_patho = None
    else:
        # binary pathology: norm -> 0, anything else -> 1
        y_patho = (info["pathology_label"] != "norm").astype(int)
        hospitals = info["hospital_id"]
        all_hospitals = None
        y = None

    out_csv = OUT_DIR / f"feature_ablation_{args.task}_{args.model}.csv"
    file_exists = out_csv.exists()
    log.info(f"Output: {out_csv.relative_to(REPO)} (exists={file_exists})")

    # ---- iterate conditions × seeds ----
    grand_t0 = time.time()
    for cond in conditions:
        if cond.startswith("sub_"):
            grp = cond.split("_", 1)[1]
            # sub_cov: N >= group_size → no real subsampling, R=1
            R = 1 if len(groups[grp]) <= args.n_features else args.n_repeats
        else:
            R = 1
        for r in range(R):
            seed = args.seed_base + r
            cols = select_columns(groups, cond, args.n_features, seed)
            label = f"{cond}  seed={seed if cond.startswith('sub_') else '-'}  n={len(cols)}"
            log.info(f"\n=== {label} ===")
            t0 = time.time()
            if args.task == "site":
                rows = run_site_cv(feats, y, cols, args.model, all_hospitals, log)
            else:
                rows = run_patho_loso(feats, y_patho, hospitals, cols, args.model, log)
            df = pd.DataFrame(rows)
            df["condition"] = cond
            df["group"] = cond.split("_", 1)[1] if "_" in cond else cond
            df["seed"] = seed
            df["n_features"] = len(cols)
            df["model"] = args.model
            df["task"] = args.task
            df.to_csv(out_csv, mode="a", header=not file_exists, index=False)
            file_exists = True
            log.info(f"    appended {len(df)} rows  (cond elapsed {time.time() - t0:.1f}s)")

    log.info(f"\nDone. Total elapsed: {(time.time() - grand_t0) / 60:.1f} min")
    log.info(f"Output: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
