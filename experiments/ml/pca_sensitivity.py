import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from catboost import CatBoostClassifier, metrics
from combatlearn.combat import ComBat

from src.harmonization.sitewise_scaler import SiteWiseStandardScaler

# --- Constants ---
INFO_FILE_PATH = 'data/ELM19/filtered/ELM19_info_filtered.csv'
FEATURES_FILE_PATH = 'data/ELM19/filtered/ELM19_features_filtered.csv'

RESULTS_PATH_SITE = 'results/tables/04_pca_sensitivity/pca_sensitivity_results_site_full.csv'
os.makedirs(os.path.dirname(RESULTS_PATH_SITE), exist_ok=True)

RESULTS_PATH_PATHO = 'results/tables/04_pca_sensitivity/pca_sensitivity_results_patho_full.csv'
os.makedirs(os.path.dirname(RESULTS_PATH_PATHO), exist_ok=True)

LOG_FILE_PATH = 'results/logs/04_pca_sensitivity/pca_sensitivity_log_full.log'
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

COVARIATES = ['age', 'gender']
SITE_COLUMN = 'hospital_id'
LABEL_COLUMN = 'pathology_label'

RANDOM_STATE = 42
N_SPLITS_SITE = 5
K_CALIBRATION = 30  # For pathology classification

#METHODS = ['neurocombat', 'covbat']
METHODS = ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
# --- PCA Parameters ---
PCA_VARIANTS = [None] #, 0.99, 0.95, 0.90, 0.80]

# --- CatBoost Parameters ---
CATBOOST_PARAMS_SITE = {
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

# Pathology params loaded from Optuna tuning (run tune_catboost_pca.py first)
CATBOOST_PARAMS_PATHO_PATH = 'config/params/catboost_patho.json'

logger = logging.getLogger(__name__)


def load_catboost_params_patho():
    """Load tuned CatBoost params from JSON file."""
    with open(CATBOOST_PARAMS_PATHO_PATH, 'r') as f:
        params = json.load(f)
    # Remove metadata keys
    params.pop('best_auc', None)
    params.pop('n_trials', None)
    # Add fixed params
    params['objective'] = 'Logloss'
    params['eval_metric'] = metrics.AUC()
    params['allow_writing_files'] = False
    params['verbose'] = False
    params['random_seed'] = RANDOM_STATE
    return params

def apply_scaling_and_pca(X_train, X_test, pca_variance, X_calib=None):
    # 1. Robust Scaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_calib_s = scaler.transform(X_calib) if X_calib is not None else None

    # 2. PCA
    #if pca_variance is not None:
    pca = PCA(n_components=pca_variance, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    X_calib_p = pca.transform(X_calib_s) if X_calib_s is not None else None
    n_comps = pca.n_components_
    # else:
    #     X_train_p = X_train_s
    #     X_test_p = X_test_s
    #     X_calib_p = X_calib_s
    #     n_comps = X_train.shape[1]

    # --- RESTORE INDICES ---
    X_train_p = pd.DataFrame(X_train_p, index=X_train.index)
    X_test_p = pd.DataFrame(X_test_p, index=X_test.index)

    if X_calib is not None:
        X_calib_p = pd.DataFrame(X_calib_p, index=X_calib.index)
    else:
        X_calib_p = None

    return X_train_p, X_test_p, X_calib_p, n_comps

def get_scores_site_classification(y_true, y_pred, hospitals):
    scores = {"mcc_overall": matthews_corrcoef(y_true, y_pred)}

    mcc_per_class = {
        f"mcc_{cls}": matthews_corrcoef((y_true == cls), (y_pred == cls)) for cls in hospitals
    }
    scores.update(mcc_per_class)
    return scores

def run_site_classification(X, y, info_df, method, pca_var):
    """
    Runs a full 5-fold stratified CV for site classification
    for a single harmonization method with PCA.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS_SITE, shuffle=True, random_state=RANDOM_STATE)
    detailed_results = []

    # Batch info
    sites = info_df[SITE_COLUMN]
    cov = info_df[COVARIATES]
    all_hospitals = sites.unique()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- STEP 1: SCALER + PCA ---
        X_train_pca, X_test_pca, _, n_features = apply_scaling_and_pca(X_train, X_test, pca_var)

        # --- STEP 2: HARMONIZATION ---
        if method == 'raw':
            X_train_harm = X_train_pca
            X_test_harm = X_test_pca
        else:
            if method == 'combat':
                harmonizer = ComBat(batch=sites, method='johnson')
            elif method == 'neurocombat':
                harmonizer = ComBat(batch=sites, discrete_covariates=cov[['gender']],
                                    continuous_covariates=cov[['age']], method='fortin')
            elif method == 'covbat':
                harmonizer = ComBat(batch=sites, discrete_covariates=cov[['gender']],
                                    continuous_covariates=cov[['age']], method='chen')
            elif method == 'sitewise':
                harmonizer = SiteWiseStandardScaler(batch=sites)
            else:
                raise ValueError(f"Method {method} not implemented in PCA script")

            harmonizer.fit(X_train_pca)

            # Transform
            X_train_harm = harmonizer.transform(X_train_pca)
            X_test_harm = harmonizer.transform(X_test_pca)

        # --- STEP 3: CLASSIFICATION ---
        clf = CatBoostClassifier(**CATBOOST_PARAMS_SITE)
        clf.fit(X_train_harm, y_train)
        preds = clf.predict(X_test_harm)

        scores_fold = get_scores_site_classification(y_test, preds, all_hospitals)
        scores_fold['method'] = method
        scores_fold['fold_id'] = fold
        scores_fold['pca_var'] = pca_var
        scores_fold['n_features'] = n_features
        detailed_results.append(scores_fold)

    return detailed_results

def get_scores_pathology_classification(y_true, y_prob, th=0.5):
    y_pred = y_prob > th

    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return mcc, acc, precision, recall, f1, auc

def run_pathology_classification(X, y, info_df, method, pca_var):
    """
    Runs LOSO CV for pathology classification with calibration subset strategy.
    Transforms features with PCA.
    """
    groups = info_df[SITE_COLUMN]
    logo = LeaveOneGroupOut()
    detailed_results = []

    for fold, (train_idx, site_idx) in enumerate(logo.split(X, y, groups)):
        hospital_test = groups.iloc[site_idx].unique()[0]

        # --- Split LOGO ---
        X_train_pool = X.iloc[train_idx]
        y_train_pool = y.iloc[train_idx]

        X_site = X.iloc[site_idx]
        y_site = y.iloc[site_idx]

        # --- Calibration Logic ---
        site_norm_mask = (y_site == 0)
        X_site_norm = X_site[site_norm_mask]
        y_site_norm = y_site[site_norm_mask]

        if len(X_site_norm) < K_CALIBRATION:
            logger.warning(
                f"Hospital {hospital_test} has fewer than {K_CALIBRATION} normal samples ({len(X_site_norm)}). Using all for calibration.")
            X_calib = X_site_norm
            # y_calib = y_site_norm
            X_test_norm = pd.DataFrame(columns=X.columns)
            y_test_norm = pd.Series(dtype=int)
        else:
            calib_idx, test_idx = train_test_split(
                X_site_norm.index, train_size=K_CALIBRATION, random_state=RANDOM_STATE
            )
            X_calib = X_site_norm.loc[calib_idx]
            X_test_norm = X_site_norm.loc[test_idx]
            y_calib = y_site_norm.loc[calib_idx]
            y_test_norm = y_site_norm.loc[test_idx]

        # Construct Final Test Set (Remaining Normals + All Pathologicals from site)
        X_test_full = pd.concat([X_test_norm, X_site[~site_norm_mask]])
        y_test_full = pd.concat([y_test_norm, y_site[~site_norm_mask]])

        # --- STEP 1: SCALER + PCA ---
        X_train_pca, X_test_pca, X_calib_pca, n_features = apply_scaling_and_pca(
            X_train_pool, X_test_full, pca_var, X_calib=X_calib
        )

        # --- STEP 2: HARMONIZATION ---
        train_norm_mask = (y_train_pool == 0)
        X_train_norm_pca = X_train_pca[train_norm_mask]

        X_fit_harm = pd.concat([X_train_norm_pca, X_calib_pca], axis=0)

        if method == 'raw':
            X_train_harm = X_train_pca
            X_test_harm = X_test_pca
        else:
            if method == 'combat':
                harmonizer = ComBat(
                    batch=info_df[SITE_COLUMN],
                    method='johnson'
                )
            elif method == 'neurocombat':
                harmonizer = ComBat(
                    batch=info_df[SITE_COLUMN],
                    discrete_covariates=info_df[['gender']],
                    continuous_covariates=info_df[['age']],
                    method='fortin'
                )
            elif method == 'covbat':
                harmonizer = ComBat(
                    batch=info_df[SITE_COLUMN],
                    discrete_covariates=info_df[['gender']],
                    continuous_covariates=info_df[['age']],
                    method='chen'
                )
            elif method == 'sitewise':
                harmonizer = SiteWiseStandardScaler(
                    batch=info_df[SITE_COLUMN]
                )
            else:
                raise ValueError(f"Method {method} not implemented in PCA script")
                harmonizer = None

            if harmonizer:
                harmonizer.fit(X_fit_harm)
                X_train_harm = harmonizer.transform(X_train_pca)
                X_test_harm = harmonizer.transform(X_test_pca)

        # --- STEP 3: CLASSIFICATION ---
        catboost_params = load_catboost_params_patho()
        clf = CatBoostClassifier(**catboost_params)
        clf.fit(X_train_harm, y_train_pool, verbose=False)

        y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]

        # Scores
        mcc, acc, precision, recall, f1, auc = get_scores_pathology_classification(y_test_full, y_pred_proba)
        scores = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1-score': f1, 'auc': auc, 'mcc': mcc,
                  'hospital': hospital_test, 'method': method, 'n_calib': len(X_calib),
                  'n_test': len(X_test_full), 'pca_var': pca_var, 'n_features': n_features}

        detailed_results.append(scores)

    return detailed_results


def main():
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Saving logs to: {LOG_FILE_PATH}")

    try:
        info = pd.read_csv(INFO_FILE_PATH)
        feats = pd.read_csv(FEATURES_FILE_PATH)

        info = info.rename(columns={
            'age_dec': 'age', 'patient_sex': 'gender',
            'institution_id': 'hospital_id', 'classification': 'pathology_label'
        })

        all_hospitals = info['hospital_id'].unique()
        info['hospital_id'] = info['hospital_id'].astype(
            pd.CategoricalDtype(categories=all_hospitals, ordered=False)
        )

        label_map = {'norm': 0, 'patho': 1, 'normal': 0, 'pathological': 1}
        y_patho = info['pathology_label'].map(label_map)
        y_site = info['hospital_id']

        logger.info(f"Loaded data. Shape: {feats.shape}")

    except FileNotFoundError:
        logger.error("Nie znaleziono plików danych.")
        return

    # --- SITE PREPARATION (Normals only) ---
    norm_mask = (y_patho == 0)
    X_site_only = feats[norm_mask].reset_index(drop=True)
    y_site_only = y_site[norm_mask].reset_index(drop=True)
    info_site_only = info[norm_mask].reset_index(drop=True)

    for target_method in METHODS:
        logger.info(f"==================================================")
        logger.info(f"--- STARTING PCA SENSITIVITY ANALYSIS (Method: {target_method}) ---")
        logger.info(f"==================================================")

        for pca_var in PCA_VARIANTS:
            var_name = f"{pca_var}" if pca_var else "No PCA"
            logger.info(f"\nProcessing PCA Variance: {var_name} ...")

            # --- 1. SITE CLASSIFICATION ---
            logger.info(f"--- STARTING EXPERIMENT: SITE CLASSIFICATION ---")
            site_results = run_site_classification(
                X_site_only, y_site_only, info_site_only, target_method, pca_var
            )

            df_results_site = pd.DataFrame(site_results)
            file_exists = os.path.isfile(RESULTS_PATH_SITE)
            df_results_site.to_csv(RESULTS_PATH_SITE, mode='a', header=not file_exists, index=False)

            mean_overall_mcc = df_results_site['mcc_overall'].mean()
            logger.info(f"Results saved for '{target_method}'. Mean Overall MCC: {mean_overall_mcc:.4f}")

            # --- 2. PATHOLOGY CLASSIFICATION ---
            logger.info(f"--- STARTING EXPERIMENT: PATHOLOGY CLASSIFICATION ---")
            patho_results = run_pathology_classification(
                feats, y_patho, info, target_method, pca_var
            )
            df_results_patho = pd.DataFrame(patho_results)
            file_exists = os.path.isfile(RESULTS_PATH_PATHO)
            df_results_patho.to_csv(RESULTS_PATH_PATHO, mode='a', header=not file_exists, index=False)

            mean_mcc = df_results_patho['mcc'].mean()
            mean_auc = df_results_patho['auc'].mean()
            logger.info(f"Results saved for '{target_method}'. Mean MCC: {mean_mcc:.4f}, Mean AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    main()