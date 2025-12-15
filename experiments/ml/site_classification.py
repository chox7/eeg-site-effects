#!/usr/bin/env python
"""
Site Classification Experiment

Runs 5-fold stratified cross-validation for site/hospital classification
using various harmonization methods.

Usage:
    # With config file (recommended)
    python experiments/ml/site_classification.py --config experiments/configs/site_classification/default.yaml

    # With config file and specific method
    python experiments/ml/site_classification.py --config experiments/configs/site_classification/default.yaml --method combat

    # Run all methods from config
    python experiments/ml/site_classification.py -c experiments/configs/site_classification/default.yaml
"""

import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from combatlearn.combat import ComBat
from src.harmonization.sitewise_scaler import SiteWiseStandardScaler
from src.harmonization.relief import RELIEFHarmonizer
from src.config import load_site_classification_config, SiteClassificationConfig

import os
import logging
import sys
import joblib


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run site classification experiment with harmonization methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with YAML config (all methods)
  python experiments/ml/site_classification.py --config experiments/configs/site_classification/default.yaml

  # Run with YAML config (specific method)
  python experiments/ml/site_classification.py --config experiments/configs/site_classification/default.yaml --method combat
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
        choices=['raw', 'sitewise', 'combat', 'neurocombat', 'covbat', 'relief'],
        help='Specific harmonization method to run (overrides config)'
    )

    parser.add_argument(
        '--features', '-f',
        type=str,
        default=None,
        help='Path to features CSV file (overrides config)'
    )

    parser.add_argument(
        '--tag', '-t',
        type=str,
        default=None,
        help='Tag to identify this run in results (e.g., filter name)'
    )

    return parser.parse_args()


def get_catboost_params(config: SiteClassificationConfig) -> dict:
    """Convert config to CatBoost parameters dictionary."""
    return config.catboost_params.to_catboost_dict(
        loss_function='MultiClass',
        eval_metric='MCC'
    )


def get_scores(y_true, y_pred, hospitals):
    """Calculate MCC scores overall and per-class."""
    scores = {"mcc_overall": matthews_corrcoef(y_true, y_pred)}
    mcc_per_class = {
        f"mcc_{cls}": matthews_corrcoef((y_true == cls), (y_pred == cls))
        for cls in hospitals
    }
    scores.update(mcc_per_class)
    return scores


def run_experiment(config: SiteClassificationConfig, harmonization_method: str = 'raw', tag: str = None):
    """
    Runs a full 5-fold stratified CV for site classification
    for a single harmonization method.

    Args:
        config: Experiment configuration
        harmonization_method: Name of harmonization method to use
        tag: Optional tag to identify this run in results
    """
    tag_str = f" [{tag}]" if tag else ""
    logger.info(f"Starting experiment: Site Classification with '{harmonization_method}'{tag_str}")

    # Create output directories
    if config.paths.pipeline_save_dir:
        os.makedirs(config.paths.pipeline_save_dir, exist_ok=True)
    if config.paths.shap_data_save_dir:
        os.makedirs(config.paths.shap_data_save_dir, exist_ok=True)

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

    y = info_df['hospital_id']
    X = features_df
    all_hospitals = np.unique(y)

    # Get CatBoost parameters from config
    catboost_params = get_catboost_params(config)

    skf = StratifiedKFold(
        n_splits=config.cv.n_splits,
        shuffle=True,
        random_state=config.cv.random_state
    )
    all_results_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"--- Method: {harmonization_method}, Fold: {fold + 1}/{config.cv.n_splits} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        sites = info_df['hospital_id']
        cov = info_df[config.data.covariates]
        pipeline_steps = []

        # Build harmonization step
        if harmonization_method == 'combat':
            pipeline_steps.append(("harmonize", ComBat(
                batch=sites,
                method='johnson'
            )))
        elif harmonization_method == 'neurocombat':
            pipeline_steps.append(("harmonize", ComBat(
                batch=sites,
                discrete_covariates=cov[['gender']],
                continuous_covariates=cov[['age']],
                method='fortin'
            )))
        elif harmonization_method == 'covbat':
            pipeline_steps.append(("harmonize", ComBat(
                batch=sites,
                discrete_covariates=cov[['gender']],
                continuous_covariates=cov[['age']],
                method='chen'
            )))
        elif harmonization_method == 'sitewise':
            pipeline_steps.append(("harmonize", SiteWiseStandardScaler(
                batch=sites
            )))

        pipeline_steps.append(("clf", CatBoostClassifier(**catboost_params)))

        # Create and Fit Pipeline
        pipeline = Pipeline(steps=pipeline_steps)

        logger.info("Fitting pipeline on training data...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating on test data...")
        preds = pipeline.predict(X_test)

        scores_fold = get_scores(y_test, preds, all_hospitals)
        scores_fold['method'] = harmonization_method
        scores_fold['fold'] = fold + 1
        if tag:
            scores_fold['tag'] = tag
        all_results_list.append(scores_fold)
        logger.info(f"Fold {fold + 1} Overall MCC: {scores_fold['mcc_overall']:.4f}")

        # Save pipeline and test data for Fold 0 (for SHAP analysis)
        if fold == 0 and config.paths.pipeline_save_dir and config.paths.shap_data_save_dir:
            logger.info("Saving pipeline and test data for Fold 0 (for SHAP analysis)...")

            tag_suffix = f"_{tag}" if tag else ""
            pipeline_filename = f"{harmonization_method}{tag_suffix}_pipeline_fold0.joblib"
            pipeline_save_path = os.path.join(config.paths.pipeline_save_dir, pipeline_filename)
            joblib.dump(pipeline, pipeline_save_path)
            logger.info(f"Pipeline saved to: {pipeline_save_path}")

            test_data_filename = f"{harmonization_method}{tag_suffix}_test_data_fold0.parquet"
            test_data_save_path = os.path.join(config.paths.shap_data_save_dir, test_data_filename)
            X_test_to_save = X_test.copy()
            X_test_to_save['y_true_hospital'] = y_test
            X_test_to_save.to_parquet(test_data_save_path)
            logger.info(f"Test data for SHAP saved to: {test_data_save_path}")

    # Save results
    df_results = pd.DataFrame(all_results_list)
    os.makedirs(os.path.dirname(config.paths.results_file), exist_ok=True)
    file_exists = os.path.isfile(config.paths.results_file)
    df_results.to_csv(config.paths.results_file, mode='a', header=not file_exists, index=False)

    mean_overall_mcc = df_results['mcc_overall'].mean()
    logger.info(f"Results saved to: {config.paths.results_file}")
    logger.info(f"Results saved for '{harmonization_method}'. Mean Overall MCC: {mean_overall_mcc:.4f}")


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
    config = load_site_classification_config(args.config)

    # Override features file if provided via CLI
    if args.features:
        config.paths.features_file = args.features
        logger.info(f"Using features file from CLI: {args.features}")

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
