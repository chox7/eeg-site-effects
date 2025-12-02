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

import os
import logging
import sys
import joblib


# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
RANDOM_STATE = 42
K_CALIBRATION = 30

INFO_FILE_PATH = 'data/ELM19/filtered/ELM19_info_filtered.csv'
FEATURES_FILE_PATH = 'data/ELM19/filtered/ELM19_features_filtered.csv'

RESULTS_PATH = 'results/tables/03_paradox_analysis/exp02_pathology_clf/exp02_pathology_clf_results.csv'

PIPELINE_SAVE_DIR = 'models/03_paradox_analysis/exp02_pathology_clf_pipelines'
SHAP_DATA_SAVE_DIR = 'results/shap_data/03_paradox_analysis/exp02_pathology_clf'

COVARIATES = ['age', 'gender']
SITE_COLUMN = 'hospital_id'
LABEL_COLUMN = 'pathology_label'

# --- CatBoost Parameters ---
CATBOOST_PARAMS = {'iterations': 700,
          'learning_rate': 0.08519504279364008,
          'depth': 6.0,
          'l2_leaf_reg': 1.1029971156522604,
          'colsample_bylevel': 0.019946626267165004,
          'objective': 'Logloss',
          'thread_count': -1,
          'boosting_type': 'Plain',
          'bootstrap_type': 'MVS',
          'eval_metric': metrics.AUC(),
          'allow_writing_files': False,
}


def get_scores(y_true, y_prob, th=0.5):
    y_pred = y_prob > th

    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return mcc, acc, precision, recall, f1, auc

def run_experiment(harmonization_method='raw'):
    """
    Runs LOSO CV for pathology classification with calibration subset strategy.
    """

    logger.info(f"Starting experiment: Pathology Classification (LOSO) with '{harmonization_method}'")

    os.makedirs(PIPELINE_SAVE_DIR, exist_ok=True)
    os.makedirs(SHAP_DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

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
    all_hospitals = info_df['hospital_id'].unique()
    info_df['hospital_id'] = info_df['hospital_id'].astype(
        pd.CategoricalDtype(categories=all_hospitals, ordered=False)
    )

    label_map = {'norm': 0, 'patho': 1, 'normal': 0, 'pathological': 1}
    y = info_df['pathology_label'].map(label_map)

    X = features_df
    groups = info_df[SITE_COLUMN]

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

        if len(X_site_norm) < K_CALIBRATION:
            logger.warning(
                f"Hospital {hospital_test} has fewer than {K_CALIBRATION} normal samples ({len(X_site_norm)}). Using all for calibration.")
            X_calib = X_site_norm
            # y_calib = y_site_norm
            X_test_norm = pd.DataFrame(columns=X.columns)  # Empty
            y_test_norm = pd.Series(dtype=int)
        else:
            X_calib, X_test_norm, y_calib, y_test_norm = train_test_split(
                X_site_norm, y_site_norm, train_size=K_CALIBRATION, random_state=RANDOM_STATE
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
        info_fit = info_df.loc[X_fit_harmonizer.index]
        if harmonization_method == 'combat':
            harmonizer = ComBat(
                batch=info_df[SITE_COLUMN],
                method='johnson'
            )
        elif harmonization_method == 'neurocombat':
            harmonizer = ComBat(
                batch=info_df[SITE_COLUMN],
                discrete_covariates=info_df[['gender']],
                continuous_covariates=info_df[['age']],
                method='fortin'
            )
        elif harmonization_method == 'covbat':
            harmonizer = ComBat(
                batch=info_df[SITE_COLUMN],
                discrete_covariates=info_df[['gender']],
                continuous_covariates=info_df[['age']],
                method='chen'
            )
        elif harmonization_method == 'sitewise':
            harmonizer = SiteWiseStandardScaler(
                batch=info_df[SITE_COLUMN]
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
        clf = GBE(esize=30, fun_model=CatBoostClassifier, **CATBOOST_PARAMS)

        clf.fit(X_train_harm, y_train_classifier, verbose=False)

        # Predict
        y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]

        # Scores
        mcc, acc, precision, recall, f1, auc = get_scores(y_test_full, y_pred_proba)
        scores = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1-score': f1, 'auc': auc, 'mcc': mcc,
                  'hospital': hospital_test, 'method': harmonization_method, 'n_calib': len(X_calib),
                  'n_test': len(X_test_full)}

        all_results_list.append(scores)
        logger.info(f"Hospital {hospital_test} - MCC: {scores['mcc']:.4f}, AUC: {scores['auc']:.4f}")

        logger.info("Saving pipeline and data for SHAP...")

        steps = []
        if harmonizer:
            steps.append(('harmonize', harmonizer))
        steps.append(('clf', clf))

        pipeline = Pipeline(steps=steps)

        # Save Pipeline
        pipeline_filename = f"{harmonization_method}_{hospital_test}_pipeline.joblib"
        joblib.dump(pipeline, os.path.join(PIPELINE_SAVE_DIR, pipeline_filename))

        # Save Test Data (Untransformed X_test_full + Labels)
        test_data_filename = f"{harmonization_method}_{hospital_test}_test_data.parquet"
        X_test_save = X_test_full.copy()
        X_test_save['y_true'] = y_test_full
        X_test_save.to_parquet(os.path.join(SHAP_DATA_SAVE_DIR, test_data_filename))

    # --- Save Results ---
    df_results = pd.DataFrame(all_results_list)
    file_exists = os.path.isfile(RESULTS_PATH)
    df_results.to_csv(RESULTS_PATH, mode='a', header=not file_exists, index=False)

    mean_mcc = df_results['mcc'].mean()
    mean_auc = df_results['auc'].mean()
    logger.info(f"Finished '{harmonization_method}'. Mean MCC: {mean_mcc:.4f}, Mean AUC: {mean_auc:.4f}")

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
        logger.error("Usage: python src/models/run_pathology_classification.py [raw|sitewise|combat|neurocombat|covbat]")
        sys.exit(1)

    run_experiment(harmonization_method=method)