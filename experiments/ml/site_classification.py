#!/usr/bin/env python
"""
Site Classification Experiment

Runs 5-fold stratified cross-validation for site/hospital classification
using various harmonization methods.

Usage:
    python experiments/ml/site_classification.py --config experiments/configs/site_classification.yaml
    python experiments/ml/site_classification.py --config experiments/configs/site_classification.yaml --method combat
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from src.harmonization import make_harmonizer
from src.utils.cv_metrics import get_scores_multiclass
from src.utils.data_prep import load_experiment_data, append_results_csv
import os
import logging
import sys
import joblib


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run site classification experiment with harmonization methods.',
    )
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--method', '-m', type=str, default=None,
                        choices=['raw', 'sitewise', 'combat', 'neurocombat', 'covbat', 'relief'],
                        help='Specific harmonization method to run (overrides config)')
    parser.add_argument('--features', '-f', type=str, default=None,
                        help='Path to features CSV file (overrides config)')
    parser.add_argument('--info', '-i', type=str, default=None,
                        help='Path to info CSV file (overrides config)')
    parser.add_argument('--tag', '-t', type=str, default=None,
                        help='Tag to identify this run in results (e.g., filter name)')
    return parser.parse_args()


def run_experiment(cfg, harmonization_method='raw', tag=None):
    tag_str = f" [{tag}]" if tag else ""
    logger.info(f"Starting experiment: Site Classification with '{harmonization_method}'{tag_str}")

    paths = cfg['paths']
    cv = cfg['cv']

    save_artifacts = cfg.get('save_artifacts', False)
    if save_artifacts:
        if paths.get('pipeline_save_dir'):
            os.makedirs(paths['pipeline_save_dir'], exist_ok=True)
        if paths.get('shap_data_save_dir'):
            os.makedirs(paths['shap_data_save_dir'], exist_ok=True)

    try:
        info_df, features_df = load_experiment_data(paths['info_file'], paths['features_file'])
        logger.info(f"  Info file: {paths['info_file']}")
        logger.info(f"  Features file: {paths['features_file']}")
    except FileNotFoundError as e:
        logger.error("Data files not found.")
        raise e

    y = info_df['hospital_id']
    X = features_df
    all_hospitals = np.unique(y)
    sites = info_df['hospital_id']
    cov = info_df[cfg['data']['covariates']]

    le = LabelEncoder()
    le.fit(all_hospitals)

    catboost_params = {
        **cfg['catboost_params'],
        'loss_function': 'MultiClass', 'eval_metric': 'MCC',
        'verbose': False, 'allow_writing_files': False,
    }

    skf = StratifiedKFold(n_splits=cv['n_splits'], shuffle=True, random_state=cv['random_state'])
    all_results_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Method: {harmonization_method}, Fold: {fold + 1}/{cv['n_splits']} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline_steps = []
        harmonizer = make_harmonizer(harmonization_method, sites, cov_df=cov)
        if harmonizer:
            pipeline_steps.append(("harmonize", harmonizer))

        pipeline_steps.append(("clf", CatBoostClassifier(**catboost_params)))
        pipeline = Pipeline(steps=pipeline_steps)

        logger.info("Fitting pipeline on training data...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating on test data...")
        y_prob = pipeline.predict_proba(X_test)

        scores_fold = get_scores_multiclass(y_test, y_prob, le)
        scores_fold['method'] = harmonization_method
        scores_fold['fold'] = fold + 1
        if tag:
            scores_fold['tag'] = tag
        all_results_list.append(scores_fold)
        logger.info(f"Fold {fold + 1} Overall MCC: {scores_fold['MCC_Overall']:.4f}")

        # Save pipeline and test data for Fold 0 (for SHAP analysis)
        if save_artifacts and fold == 0 and paths.get('pipeline_save_dir') and paths.get('shap_data_save_dir'):
            logger.info("Saving pipeline and test data for Fold 0 (for SHAP analysis)...")
            tag_suffix = f"_{tag}" if tag else ""
            joblib.dump(pipeline, os.path.join(
                paths['pipeline_save_dir'],
                f"{harmonization_method}{tag_suffix}_pipeline_fold0.joblib"
            ))
            X_test_to_save = X_test.copy()
            X_test_to_save['y_true_hospital'] = y_test
            X_test_to_save.to_parquet(os.path.join(
                paths['shap_data_save_dir'],
                f"{harmonization_method}{tag_suffix}_test_data_fold0.parquet"
            ))

    df_results = pd.DataFrame(all_results_list)
    append_results_csv(df_results, paths['results_file'])

    logger.info(f"Results saved to: {paths['results_file']}")
    logger.info(f"Results saved for '{harmonization_method}'. Mean Overall MCC: {df_results['MCC_Overall'].mean():.4f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.features:
        cfg['paths']['features_file'] = args.features
        logger.info(f"Using features file from CLI: {args.features}")
    if args.info:
        cfg['paths']['info_file'] = args.info
        logger.info(f"Using info file from CLI: {args.info}")

    if cfg.get('experiment_name'):
        logger.info(f"Running experiment: {cfg['experiment_name']}")

    methods_to_run = [args.method] if args.method else cfg['harmonization_methods']

    valid_methods = ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
    for method in methods_to_run:
        if method not in valid_methods:
            logger.error(f"Unknown method '{method}'. Valid: {valid_methods}")
            sys.exit(1)

    for method in methods_to_run:
        run_experiment(cfg, harmonization_method=method, tag=args.tag)


if __name__ == '__main__':
    main()