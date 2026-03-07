import pandas as pd
import os
import sys
import yaml
import logging
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, metrics as catboost_metrics

from src.harmonization import make_harmonizer
from src.utils.cv_metrics import get_scores_binary, get_scores_multiclass
from src.utils.data_prep import load_experiment_data, prepare_pathology_labels, append_results_csv

CONFIG_PATH = 'experiments/configs/pca_sensitivity.yaml'

# --- Constants ---
INFO_FILE_PATH = 'data/ELM19/filtered/ELM19_info_filtered.csv'
FEATURES_FILE_PATH = 'data/ELM19/filtered/ELM19_features_filtered.csv'

RESULTS_PATH_SITE = 'results/tables/05_pca_sensitivity/pca_sensitivity_results_site_full.csv'
os.makedirs(os.path.dirname(RESULTS_PATH_SITE), exist_ok=True)

RESULTS_PATH_PATHO = 'results/tables/05_pca_sensitivity/pca_sensitivity_results_patho_full_single_catboost.csv'
os.makedirs(os.path.dirname(RESULTS_PATH_PATHO), exist_ok=True)

RESULTS_PATH_PATHO_MULTIMODEL = (
    'results/tables/05_pca_sensitivity/pca_sensitivity_results_patho_multimodel.csv'
)

LOG_FILE_PATH = 'results/logs/05_pca_sensitivity/pca_sensitivity_log_full_single_catboost.log'
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

COVARIATES = ['age', 'gender']
SITE_COLUMN = 'hospital_id'
LABEL_COLUMN = 'pathology_label'

RANDOM_STATE = 42
N_SPLITS_SITE = 5
K_CALIBRATION = 30  # For pathology classification
N_PARALLEL = 26

METHODS = ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
MODELS = ['logreg', 'svm', 'catboost']
# --- PCA Parameters ---
PCA_VARIANTS = ['none', 'all', 0.99, 0.95, 0.90, 0.80]

logger = logging.getLogger(__name__)


def apply_scaling_and_pca(X_train, X_test, pca_variance, X_calib=None):
    # 1. Robust Scaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_calib_s = scaler.transform(X_calib) if X_calib is not None else None

    # 2. PCA (or skip)
    if pca_variance == 'none':
        # No PCA - just return scaled data
        X_train_p = X_train_s
        X_test_p = X_test_s
        X_calib_p = X_calib_s
        n_comps = X_train.shape[1]
    else:
        # PCA: 'all' → None (all components), float → variance threshold
        n_components = None if pca_variance == 'all' else pca_variance
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        X_calib_p = pca.transform(X_calib_s) if X_calib_s is not None else None
        n_comps = pca.n_components_

    # Restore indices
    X_train_p = pd.DataFrame(X_train_p, index=X_train.index)
    X_test_p = pd.DataFrame(X_test_p, index=X_test.index)
    if X_calib is not None:
        X_calib_p = pd.DataFrame(X_calib_p, index=X_calib.index)
    else:
        X_calib_p = None

    return X_train_p, X_test_p, X_calib_p, n_comps

def run_site_classification(X, y, info_df, method, pca_var, catboost_params, model_name='catboost'):
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

    le = LabelEncoder()
    le.fit(all_hospitals)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- STEP 1: SCALER + PCA ---
        X_train_pca, X_test_pca, _, n_features = apply_scaling_and_pca(X_train, X_test, pca_var)

        # --- STEP 2: HARMONIZATION ---
        harmonizer = make_harmonizer(method, sites, cov_df=cov)
        if harmonizer:
            harmonizer.fit(X_train_pca)
            X_train_harm = harmonizer.transform(X_train_pca)
            X_test_harm = harmonizer.transform(X_test_pca)
        else:
            X_train_harm = X_train_pca
            X_test_harm = X_test_pca

        # --- STEP 3: CLASSIFICATION ---
        if model_name == 'catboost':
            clf = CatBoostClassifier(**catboost_params)
            clf.fit(X_train_harm, y_train, verbose=False)
        elif model_name == 'svm':
            clf = SVC(kernel='rbf', probability=True, C=1.0, random_state=RANDOM_STATE)
            clf.fit(X_train_harm, y_train)
        elif model_name == 'logreg':
            clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, C=1.0)
            clf.fit(X_train_harm, y_train)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_prob = clf.predict_proba(X_test_harm)

        scores_fold = get_scores_multiclass(y_test, y_prob, le)
        scores_fold['model'] = model_name
        scores_fold['method'] = method
        scores_fold['fold_id'] = fold
        scores_fold['pca_var'] = pca_var
        scores_fold['n_features'] = n_features
        detailed_results.append(scores_fold)

    return detailed_results

def run_pathology_classification(X, y, info_df, method, pca_var, catboost_params, model_name='catboost'):
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

        harmonizer = make_harmonizer(method, info_df[SITE_COLUMN], cov_df=info_df)
        if harmonizer:
            harmonizer.fit(X_fit_harm)
            X_train_harm = harmonizer.transform(X_train_pca)
            X_test_harm = harmonizer.transform(X_test_pca)
        else:
            X_train_harm = X_train_pca
            X_test_harm = X_test_pca

        # --- STEP 3: CLASSIFICATION ---
        if model_name == 'catboost':
            clf = CatBoostClassifier(**catboost_params)
            clf.fit(X_train_harm, y_train_pool, verbose=False)
            y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]
        elif model_name == 'svm':
            clf = SVC(kernel='rbf', probability=True, C=1.0, random_state=RANDOM_STATE)
            clf.fit(X_train_harm, y_train_pool)
            y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]
        elif model_name == 'logreg':
            clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, C=1.0)
            clf.fit(X_train_harm, y_train_pool)
            y_pred_proba = clf.predict_proba(X_test_harm)[:, 1]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Scores
        s = get_scores_binary(y_test_full, y_pred_proba)
        scores = {
            'mcc': s['MCC'], 'accuracy': s['Accuracy'], 'precision': s['Precision'],
            'recall': s['Recall'], 'f1-score': s['F1-Score'], 'auc': s['AUC'],
            'model': model_name, 'hospital': hospital_test, 'method': method,
            'n_calib': len(X_calib), 'n_test': len(X_test_full),
            'pca_var': pca_var, 'n_features': n_features,
        }

        detailed_results.append(scores)

    return detailed_results


def _run_job(task, X, y, info_df, method, pca_var, catboost_params, model_name):
    """Run a single (task, method, pca_var, model) combination. Used by joblib."""
    if task == 'site':
        results = run_site_classification(
            X, y, info_df, method, pca_var, catboost_params, model_name,
        )
    else:
        results = run_pathology_classification(
            X, y, info_df, method, pca_var, catboost_params, model_name,
        )
    return task, method, pca_var, model_name, results


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

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    catboost_params_site = {
        **cfg['catboost_params_site'],
        'loss_function': 'MultiClass', 'eval_metric': 'MCC',
        'verbose': False, 'allow_writing_files': False,
        'task_type': 'CPU', 'thread_count': 1,
    }
    catboost_params_patho = {
        **cfg['catboost_params_patho'],
        'loss_function': 'Logloss', 'eval_metric': catboost_metrics.AUC(),
        'verbose': False, 'allow_writing_files': False,
        'task_type': 'CPU', 'thread_count': 1,
    }
    logger.info(f"Loaded config from {CONFIG_PATH}")

    try:
        info, feats = load_experiment_data(INFO_FILE_PATH, FEATURES_FILE_PATH)
        y_patho = prepare_pathology_labels(info)
        y_site = info['hospital_id']
        logger.info(f"Loaded data. Shape: {feats.shape}")
    except FileNotFoundError:
        logger.error("Data files not found.")
        return

    # --- SITE PREPARATION (Normals only) ---
    norm_mask = (y_patho == 0)
    X_site_only = feats[norm_mask].reset_index(drop=True)
    y_site_only = y_site[norm_mask].reset_index(drop=True)
    info_site_only = info[norm_mask].reset_index(drop=True)

    job_args = []
    for method in METHODS:
        for pca_var in PCA_VARIANTS:
            job_args.append(('site', X_site_only, y_site_only, info_site_only,
                             method, pca_var, catboost_params_site))
            job_args.append(('patho', feats, y_patho, info,
                             method, pca_var, catboost_params_patho))

    for model_name in MODELS:
        logger.info(f"=== Starting model group: {model_name} ===")

        # SVM is single-threaded — parallelize jobs across cores
        if model_name == 'svm':
            logger.info(f"Running {len(job_args)} {model_name} jobs with {N_PARALLEL} parallel workers...")
            jobs = [delayed(_run_job)(*args, model_name) for args in job_args]
            results = Parallel(n_jobs=N_PARALLEL, verbose=10, prefer='threads')(jobs)
        else:
            logger.info(f"Running {len(job_args)} {model_name} jobs sequentially...")
            results = [_run_job(*args, model_name) for args in job_args]

        # Collect and save after each model group
        site_results = []
        patho_results = []
        for task, method, pca_var, _, result_list in results:
            if task == 'site':
                site_results.extend(result_list)
            else:
                patho_results.extend(result_list)

        if site_results:
            append_results_csv(pd.DataFrame(site_results), RESULTS_PATH_SITE)
        if patho_results:
            append_results_csv(pd.DataFrame(patho_results), RESULTS_PATH_PATHO_MULTIMODEL)

        logger.info(f"=== Finished model group: {model_name} — results saved ===")

    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()