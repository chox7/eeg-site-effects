"""
Optuna hyperparameter tuning for CatBoost pathology classification.

Usage:
    python experiments/ml/tune_catboost_pca.py [--n_trials 100]
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import logging
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# --- Constants ---
INFO_FILE_PATH = 'data/ELM19/filtered/ELM19_info_filtered.csv'
FEATURES_FILE_PATH = 'data/ELM19/filtered/ELM19_features_filtered.csv'
RESULTS_DIR = 'results/logs/04_pca_sensitivity/tuning'
PARAMS_DIR = 'config/params'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS = 3

logger = logging.getLogger(__name__)


def create_objective(X, y):
    """
    Creates an Optuna objective function for CatBoost tuning.

    Uses stratified K-fold CV on raw (scaled) features.
    Optimizes for AUC on binary pathology classification.
    """
    def objective(trial):
        # --- Hyperparameter search space ---
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'objective': 'Logloss',
            'max_bin': 32,
            'random_seed': RANDOM_STATE,
            'verbose': False,
            'allow_writing_files': False,
            'task_type': 'GPU',
        }

        # --- Cross-validation ---
        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Train model
            clf = CatBoostClassifier(**params)
            clf.fit(X_train_s, y_train, eval_set=(X_val_s, y_val), early_stopping_rounds=50, verbose=False)

            # Evaluate
            y_pred_proba = clf.predict_proba(X_val_s)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    return objective


def main():
    parser = argparse.ArgumentParser(description='Tune CatBoost for pathology classification (for PCA experiment)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(RESULTS_DIR, 'tuning.log'), mode='w')
        ]
    )

    logger.info(f"Starting hyperparameter tuning with {args.n_trials} trials")
    logger.info("Tuning on raw features (no PCA) - params will be used for PCA comparison")

    # Load data
    info = pd.read_csv(INFO_FILE_PATH)
    feats = pd.read_csv(FEATURES_FILE_PATH)

    info = info.rename(columns={
        'age_dec': 'age', 'patient_sex': 'gender',
        'institution_id': 'hospital_id', 'classification': 'pathology_label'
    })

    label_map = {'norm': 0, 'patho': 1, 'normal': 0, 'pathological': 1}
    y = info['pathology_label'].map(label_map)
    X = feats

    logger.info(f"Data loaded. Shape: {X.shape}, Class distribution: {y.value_counts().to_dict()}")

    # Run Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    objective = create_objective(X, y)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    logger.info(f"\n{'='*50}")
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save best params to shared config folder
    output_file = os.path.join(PARAMS_DIR, 'catboost_patho.json')

    best_params = study.best_params.copy()
    best_params['best_auc'] = study.best_value
    best_params['n_trials'] = args.n_trials

    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"\nBest parameters saved to: {output_file}")

    # Also save study history
    history_file = os.path.join(RESULTS_DIR, 'study_history.csv')
    trials_df = study.trials_dataframe()
    trials_df.to_csv(history_file, index=False)
    logger.info(f"Study history saved to: {history_file}")


if __name__ == "__main__":
    main()
