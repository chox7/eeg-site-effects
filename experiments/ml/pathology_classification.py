#!/usr/bin/env python
"""
Pathology Classification Experiment

Runs Leave-One-Site-Out cross-validation for pathology classification
with a calibration subset strategy.

Usage:
    # With config file (recommended)
    python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification/default.yaml

    # With config file and specific method
    python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification/default.yaml --method combat

    # Run all methods from config
    python experiments/ml/pathology_classification.py -c experiments/configs/pathology_classification/default.yaml
"""

import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, metrics
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from combatlearn.combat import ComBat
from src.harmonization.sitewise_scaler import SiteWiseStandardScaler
from src.harmonization.relief import RELIEFHarmonizer
from src.models.gbe import GBE
from src.config import load_pathology_classification_config, PathologyClassificationConfig

import os
import logging
import sys
import joblib


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run pathology classification experiment with harmonization methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with YAML config (all methods)
  python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification/default.yaml

  # Run with YAML config (specific method)
  python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification/default.yaml --method combat
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--method', '-m',
        type=str,
        default=None,
        choices=['raw', 'sitewise', 'combat', 'neurocombat', 'covbat'],
        help='Specific harmonization method to run (overrides config)'
    )

    parser.add_argument(
        '--features', '-f',
        type=str,
        default=None,
        help='Path to features CSV file (overrides config)'
    )

    parser.add_argument(
        '--info', '-i',
        type=str,
        default=None,
        help='Path to info CSV file (overrides config)'
    )

    parser.add_argument(
        '--tag', '-t',
        type=str,
        default=None,
        help='Tag to identify this run in results (e.g., filter name)'
    )

    return parser.parse_args()


def get_catboost_params(config: PathologyClassificationConfig) -> dict:
    """Convert config to CatBoost parameters dictionary."""
    return config.catboost_params.to_catboost_dict(
        loss_function='Logloss',
        eval_metric=metrics.AUC()
    )


def get_scores(y_true, y_prob, th=0.5):
    """Calculate classification scores."""
    y_pred = y_prob > th

    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return mcc, acc, precision, recall, f1, auc


def run_experiment(config: PathologyClassificationConfig, harmonization_method: str = 'raw', tag: str = None):
    """
    Runs LOSO CV for pathology classification with calibration subset strategy.

    Args:
        config: Experiment configuration
        harmonization_method: Name of harmonization method to use
        tag: Optional tag to identify this run in results
    """
    tag_str = f" [{tag}]" if tag else ""
    logger.info(f"Starting experiment: Pathology Classification (LOSO) with '{harmonization_method}'{tag_str}")

    # Create output directories
    if config.paths.pipeline_save_dir:
        os.makedirs(config.paths.pipeline_save_dir, exist_ok=True)
    if config.paths.shap_data_save_dir:
        os.makedirs(config.paths.shap_data_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.paths.results_file), exist_ok=True)

    # Load data
    try:
        info_df = pd.read_csv(config.paths.info_file)
        features_df = pd.read_csv(config.paths.features_file)
        logger.info("Data files loaded successfully.")
        logger.info(f"  Info file: {config.paths.info_file}")
        logger.info(f"  Features file: {config.paths.features_file}")
    except FileNotFoundError as e:
        logger.error(f"Error: Data files not found.")
        logger.error(f"Checked: {config.paths.info_file}")
        logger.error(f"Checked: {config.paths.features_file}")
        raise e

    # Rename columns to standard names
    info_df = info_df.rename(columns={
        'age_dec': 'age',
        'patient_sex': 'gender',
        'institution_id': 'hospital_id',
        'classification': 'pathology_label'
    })
    all_hospitals = info_df['hospital_id'].unique()
    info_df['hospital_id'] = info_df['hospital_id'].astype(
        pd.CategoricalDtype(categories=all_hospitals, ordered=False)
    )

    label_map = {'norm': 0, 'patho': 1, 'normal': 0, 'pathological': 1}
    y = info_df['pathology_label'].map(label_map)

    X = features_df
    groups = info_df['hospital_id']

    # Get CatBoost parameters from config
    catboost_params = get_catboost_params(config)

    logo = LeaveOneGroupOut()
    all_results_list = []

    for fold, (train_idx, site_idx) in enumerate(logo.split(X, y, groups)):
        # Identify Test Hospital
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

        k_calibration = config.cv.k_calibration
        if len(X_site_norm) < k_calibration:
            logger.warning(
                f"Hospital {hospital_test} has fewer than {k_calibration} normal samples ({len(X_site_norm)}). Using all for calibration.")
            X_calib = X_site_norm
            X_test_norm = pd.DataFrame(columns=X.columns)  # Empty
            y_test_norm = pd.Series(dtype=int)
        else:
            X_calib, X_test_norm, y_calib, y_test_norm = train_test_split(
                X_site_norm, y_site_norm, train_size=k_calibration, random_state=config.cv.random_state
            )

        # Construct Fit Set for Harmonizer (Train Normals + Calibration Normals)
        train_norm_mask = (y_train_pool == 0)
        X_train_norm = X_train_pool[train_norm_mask]

        X_fit_harmonizer = pd.concat([X_train_norm, X_calib], axis=0)

        # Construct Final Test Set (Remaining Normals + All Pathologicals from site)
        X_test_full = pd.concat([X_test_norm, X_site[~site_norm_mask]])
        y_test_full = pd.concat([y_test_norm, y_site[~site_norm_mask]])

        # Construct Final Train Set for Classifier
        X_train_classifier = X_train_pool
        y_train_classifier = y_train_pool

        # --- 3. Harmonization ---
        harmonizer = None
        if harmonization_method == 'combat':
            harmonizer = ComBat(
                batch=info_df['hospital_id'],
                method='johnson'
            )
        elif harmonization_method == 'neurocombat':
            harmonizer = ComBat(
                batch=info_df['hospital_id'],
                discrete_covariates=info_df[['gender']],
                continuous_covariates=info_df[['age']],
                method='fortin'
            )
        elif harmonization_method == 'covbat':
            harmonizer = ComBat(
                batch=info_df['hospital_id'],
                discrete_covariates=info_df[['gender']],
                continuous_covariates=info_df[['age']],
                method='chen'
            )
        elif harmonization_method == 'sitewise':
            harmonizer = SiteWiseStandardScaler(
                batch=info_df['hospital_id']
            )

        if harmonizer:
            logger.info("Fitting harmonizer (on Normals + Calibration)...")
            harmonizer.fit(X_fit_harmonizer)

            logger.info("Transforming train and test data...")
            X_train_harm = harmonizer.transform(X_train_classifier)
            X_test_harm = harmonizer.transform(X_test_full)
        else:
            # Raw
            X_train_harm = X_train_classifier
            X_test_harm = X_test_full

        # --- 4. Classification ---
        logger.info("Training classifier...")
        clf = GBE(esize=config.ensemble_size, fun_model=CatBoostClassifier, **catboost_params)

        clf.fit(X_train_harm, y_train_classifier, verbose=False)

        # Predict
        y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]

        # Scores
        mcc, acc, precision, recall, f1, auc = get_scores(y_test_full, y_pred_proba)
        scores = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1-score': f1, 'auc': auc, 'mcc': mcc,
                  'hospital': hospital_test, 'method': harmonization_method, 'n_calib': len(X_calib),
                  'n_test': len(X_test_full)}
        if tag:
            scores['tag'] = tag

        all_results_list.append(scores)
        logger.info(f"Hospital {hospital_test} - MCC: {scores['mcc']:.4f}, AUC: {scores['auc']:.4f}")

        # Save pipeline and data for SHAP
        if config.paths.pipeline_save_dir and config.paths.shap_data_save_dir:
            logger.info("Saving pipeline and data for SHAP...")

            steps = []
            if harmonizer:
                steps.append(('harmonize', harmonizer))
            steps.append(('clf', clf))

            pipeline = Pipeline(steps=steps)

            # Save Pipeline
            tag_suffix = f"_{tag}" if tag else ""
            pipeline_filename = f"{harmonization_method}_{hospital_test}{tag_suffix}_pipeline.joblib"
            joblib.dump(pipeline, os.path.join(config.paths.pipeline_save_dir, pipeline_filename))

            # Save Test Data (Untransformed X_test_full + Labels)
            test_data_filename = f"{harmonization_method}_{hospital_test}{tag_suffix}_test_data.parquet"
            X_test_save = X_test_full.copy()
            X_test_save['y_true'] = y_test_full
            X_test_save.to_parquet(os.path.join(config.paths.shap_data_save_dir, test_data_filename))

    # --- Save Results ---
    df_results = pd.DataFrame(all_results_list)
    file_exists = os.path.isfile(config.paths.results_file)
    df_results.to_csv(config.paths.results_file, mode='a', header=not file_exists, index=False)

    mean_mcc = df_results['mcc'].mean()
    mean_auc = df_results['auc'].mean()
    logger.info(f"Results saved to: {config.paths.results_file}")
    logger.info(f"Finished '{harmonization_method}'. Mean MCC: {mean_mcc:.4f}, Mean AUC: {mean_auc:.4f}")


def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    args = parse_args()

    # Load configuration
    config = load_pathology_classification_config(args.config)

    # Override paths if provided via CLI
    if args.features:
        config.paths.features_file = args.features
        logger.info(f"Using features file from CLI: {args.features}")

    if args.info:
        config.paths.info_file = args.info
        logger.info(f"Using info file from CLI: {args.info}")

    if config.experiment_name:
        logger.info(f"Running experiment: {config.experiment_name}")

    # Determine which methods to run
    if args.method:
        # CLI --method flag takes precedence
        methods_to_run = [args.method]
    else:
        # Run all methods from config
        methods_to_run = config.harmonization_methods

    # Validate methods
    valid_methods = ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
    for method in methods_to_run:
        if method not in valid_methods:
            logger.error(f"Error: Unknown method '{method}'.")
            logger.error(f"Valid methods: {valid_methods}")
            sys.exit(1)

    # Run experiments
    for method in methods_to_run:
        run_experiment(config, harmonization_method=method, tag=args.tag)


if __name__ == '__main__':
    main()
