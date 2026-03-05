#!/usr/bin/env python
"""
Pathology Classification Experiment

Runs Leave-One-Site-Out cross-validation for pathology classification
with a calibration subset strategy.

Usage:
    python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification.yaml
    python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification.yaml --method combat
"""

import argparse
import yaml
import pandas as pd
from catboost import CatBoostClassifier, metrics as catboost_metrics
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from src.harmonization import make_harmonizer
from src.models.gbe import GBE
from src.utils.cv_metrics import get_scores_binary
from src.utils.data_prep import load_experiment_data, prepare_pathology_labels, append_results_csv

import os
import logging
import sys
import joblib


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run pathology classification experiment with harmonization methods.',
    )
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--method', '-m', type=str, default=None,
                        choices=['raw', 'sitewise', 'combat', 'neurocombat', 'covbat'],
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
    logger.info(f"Starting experiment: Pathology Classification (LOSO) with '{harmonization_method}'{tag_str}")

    paths = cfg['paths']
    cv = cfg['cv']

    save_artifacts = cfg.get('save_artifacts', False)
    if save_artifacts:
        if paths.get('pipeline_save_dir'):
            os.makedirs(paths['pipeline_save_dir'], exist_ok=True)
        if paths.get('shap_data_save_dir'):
            os.makedirs(paths['shap_data_save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(paths['results_file']), exist_ok=True)

    try:
        info_df, features_df = load_experiment_data(paths['info_file'], paths['features_file'])
        logger.info(f"  Info file: {paths['info_file']}")
        logger.info(f"  Features file: {paths['features_file']}")
    except FileNotFoundError as e:
        logger.error("Data files not found.")
        raise e

    y = prepare_pathology_labels(info_df)
    X = features_df
    groups = info_df['hospital_id']

    catboost_params = {
        **cfg['catboost_params'],
        'loss_function': 'Logloss', 'eval_metric': catboost_metrics.AUC(),
        'verbose': False, 'allow_writing_files': False,
    }
    k_calibration = cv['k_calibration']
    ensemble_size = cfg['ensemble_size']

    logo = LeaveOneGroupOut()
    all_results_list = []

    for fold, (train_idx, site_idx) in enumerate(logo.split(X, y, groups)):
        hospital_test = groups.iloc[site_idx].unique()[0]
        logger.info(f"--- Fold {fold + 1}: Holding out hospital '{hospital_test}' ---")

        # --- 1. Split Data ---
        # Training pool (all other sites)
        X_train_pool = X.iloc[train_idx]
        y_train_pool = y.iloc[train_idx]

        # Test site data
        X_site = X.iloc[site_idx]
        y_site = y.iloc[site_idx]

        # --- 2. Calibration Logic ---
        # Find normals in the test site
        site_norm_mask = (y_site == 0)  # 0 is normal
        X_site_norm = X_site[site_norm_mask]
        y_site_norm = y_site[site_norm_mask]

        if len(X_site_norm) < k_calibration:
            logger.warning(
                f"Hospital {hospital_test} has fewer than {k_calibration} normal samples "
                f"({len(X_site_norm)}). Using all for calibration."
            )
            X_calib = X_site_norm
            X_test_norm = pd.DataFrame(columns=X.columns)  # Empty
            y_test_norm = pd.Series(dtype=int)
        else:
            X_calib, X_test_norm, y_calib, y_test_norm = train_test_split(
                X_site_norm, y_site_norm,
                train_size=k_calibration, random_state=cv['random_state']
            )

        # Construct Fit Set for Harmonizer (Train Normals + Calibration Normals)
        X_fit_harmonizer = pd.concat([X_train_pool[y_train_pool == 0], X_calib], axis=0)

        # Construct Final Test Set (Remaining Normals + All Pathologicals from site)
        X_test_full = pd.concat([X_test_norm, X_site[~site_norm_mask]])
        y_test_full = pd.concat([y_test_norm, y_site[~site_norm_mask]])

        # Construct Final Train Set for Classifier
        X_train_classifier = X_train_pool
        y_train_classifier = y_train_pool

        # --- 3. Harmonization ---
        harmonizer = make_harmonizer(
            harmonization_method, info_df['hospital_id'], cov_df=info_df
        )

        if harmonizer:
            logger.info("Fitting harmonizer (on Normals + Calibration)...")
            harmonizer.fit(X_fit_harmonizer)
            X_train_harm = harmonizer.transform(X_train_classifier)
            X_test_harm = harmonizer.transform(X_test_full)
        else:
            X_train_harm = X_train_classifier
            X_test_harm = X_test_full

        # --- 4. Classification ---
        logger.info("Training classifier...")
        clf = GBE(esize=ensemble_size, fun_model=CatBoostClassifier, **catboost_params)
        clf.fit(X_train_harm, y_train_classifier, verbose=False)
        y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]

        s = get_scores_binary(y_test_full, y_pred_proba)
        scores = {
            'mcc': s['MCC'], 'accuracy': s['Accuracy'], 'precision': s['Precision'],
            'recall': s['Recall'], 'f1-score': s['F1-Score'], 'auc': s['AUC'],
            'hospital': hospital_test, 'method': harmonization_method,
            'n_calib': len(X_calib), 'n_test': len(X_test_full),
        }
        if tag:
            scores['tag'] = tag

        all_results_list.append(scores)
        logger.info(f"Hospital {hospital_test} - MCC: {scores['mcc']:.4f}, AUC: {scores['auc']:.4f}")

        if save_artifacts and paths.get('pipeline_save_dir') and paths.get('shap_data_save_dir'):
            logger.info("Saving pipeline and data for SHAP...")
            steps = [('harmonize', harmonizer)] if harmonizer else []
            steps.append(('clf', clf))
            pipeline = Pipeline(steps=steps)

            # Save Pipeline
            tag_suffix = f"_{tag}" if tag else ""
            joblib.dump(pipeline, os.path.join(
                paths['pipeline_save_dir'],
                f"{harmonization_method}_{hospital_test}{tag_suffix}_pipeline.joblib"
            ))
            X_test_save = X_test_full.copy()
            X_test_save['y_true'] = y_test_full
            X_test_save.to_parquet(os.path.join(
                paths['shap_data_save_dir'],
                f"{harmonization_method}_{hospital_test}{tag_suffix}_test_data.parquet"
            ))

    # --- Save Results ---
    df_results = pd.DataFrame(all_results_list)
    append_results_csv(df_results, paths['results_file'])

    logger.info(f"Results saved to: {paths['results_file']}")
    logger.info(
        f"Finished '{harmonization_method}'. "
        f"Mean MCC: {df_results['mcc'].mean():.4f}, Mean AUC: {df_results['auc'].mean():.4f}"
    )


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
