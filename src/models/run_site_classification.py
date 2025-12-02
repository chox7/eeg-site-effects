import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from combatlearn.combat import ComBat
from src.harmonization.sitewise_scaler import SiteWiseStandardScaler
from src.harmonization.relief import RELIEFHarmonizer

import os
import logging
import sys
import joblib


# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
N_SPLITS = 5
RANDOM_STATE = 42

INFO_FILE_PATH = 'data/ELM19/filtered/ELM19_info_filtered_norm.csv'
FEATURES_FILE_PATH = 'data/ELM19/filtered/ELM19_features_filtered_norm.csv'

RESULTS_PATH = 'results/tables/03_paradox_analysis/exp01_site_clf/exp01_site_clf_results.csv'

PIPELINE_SAVE_DIR= 'models/03_paradox_analysis/exp01_site_clf_pipelines'
SHAP_DATA_SAVE_DIR = 'results/shap_data/03_paradox_analysis/exp01_site_clf'

COVARIATES = ['age', 'gender']
SITE_COLUMN = 'hospital_id'

# --- CatBoost Parameters ---
CATBOOST_PARAMS = {
    'iterations': 2000,
    'learning_rate': 0.2136106733298358,
    'depth': 5.0,
    'l2_leaf_reg': 1.0050061307458207,
    'early_stopping_rounds': 50,

    'loss_function': 'MultiClass',
    'eval_metric': 'MCC',

    'task_type': "GPU",
    'thread_count': 20,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'allow_writing_files': False
}

def get_scores(y_true, y_pred, hospitals):
    scores = {"mcc_overall": matthews_corrcoef(y_true, y_pred)}

    mcc_per_class = {
        f"mcc_{cls}": matthews_corrcoef((y_true == cls), (y_pred == cls)) for cls in hospitals
    }
    scores.update(mcc_per_class)
    return scores

def run_experiment(harmonization_method='raw'):
    """
    Runs a full 5-fold stratified CV for site classification
    for a single harmonization method.
    """

    logger.info(f"Starting experiment: Site Classification with '{harmonization_method}'")

    os.makedirs(PIPELINE_SAVE_DIR, exist_ok=True)
    os.makedirs(SHAP_DATA_SAVE_DIR, exist_ok=True)

    try:
        info_df = pd.read_csv(INFO_FILE_PATH)
        features_df = pd.read_csv(FEATURES_FILE_PATH)
        logger.info("Data files loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: Data files not found.")
        logger.error(f"Checked: {INFO_FILE_PATH}")
        logger.error(f"Checked: {FEATURES_FILE_PATH}")
        return

    info_df = info_df.rename(columns={
        'age_dec': 'age',
        'patient_sex': 'gender',
        'institution_id': 'hospital_id',
        'classification': 'pathology_label'
    })

    y = info_df[SITE_COLUMN]
    X = features_df
    all_hospitals = np.unique(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        logger.info(f"--- Method: {harmonization_method}, Fold: {fold + 1}/{N_SPLITS} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        sites= info_df[SITE_COLUMN]
        cov = info_df[COVARIATES]
        pipeline_steps = []
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
        elif harmonization_method == 'relief':
            pipeline_steps.append(("harmonize", RELIEFHarmonizer(
                batch=sites,
                mod=cov,
                scale_features=True,
                eps=1e-3,
                max_iter=1000,
                verbose=True
            )))


        pipeline_steps.append(("clf", CatBoostClassifier(**CATBOOST_PARAMS)))

        # --- Create and Fit Pipeline ---
        pipeline = Pipeline(steps=pipeline_steps)

        logger.info("Fitting pipeline on training data...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating on test data...")
        preds = pipeline.predict(X_test)

        scores_fold = get_scores(y_test, preds, all_hospitals)
        scores_fold['method'] = harmonization_method
        scores_fold['fold'] = fold + 1
        all_results_list.append(scores_fold)
        logger.info(f"Fold {fold + 1} Overall MCC: {scores_fold['mcc_overall']:.4f}")

        if fold == 0:
            logger.info("Saving pipeline and test data for Fold 0 (for SHAP analysis)...")

            # 1. Save the pipeline object
            pipeline_filename = f"{harmonization_method}_pipeline_fold0.joblib"
            pipeline_save_path = os.path.join(PIPELINE_SAVE_DIR, pipeline_filename)
            joblib.dump(pipeline, pipeline_save_path)
            logger.info(f"Pipeline saved to: {pipeline_save_path}")

            # 2. Save the corresponding untransformed X_test and y_test
            test_data_filename = f"{harmonization_method}_test_data_fold0.parquet"
            test_data_save_path = os.path.join(SHAP_DATA_SAVE_DIR, test_data_filename)
            X_test_to_save = X_test.copy()
            X_test_to_save['y_true_hospital'] = y_test  # Add labels for context
            X_test_to_save.to_parquet(test_data_save_path)
            logger.info(f"Test data for SHAP saved to: {test_data_save_path}")

    # Save results
    df_results = pd.DataFrame(all_results_list)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_PATH)
    df_results.to_csv(RESULTS_PATH, mode='a', header=not file_exists, index=False)

    mean_overall_mcc = df_results['mcc_overall'].mean()
    logger.info(f"Results saved for '{harmonization_method}'. Mean Overall MCC: {mean_overall_mcc:.4f}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)

    if len(sys.argv) > 1:
        method = sys.argv[1]
    else:
        method = 'raw'

    if method not in ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']:
        logger.error(f"Error: Unknown method '{method}'.")
        logger.error("Usage: python src/models/run_site_classification.py [raw|sitewise|combat|neurocombat|covbat]")
        sys.exit(1)

    run_experiment(harmonization_method=method)