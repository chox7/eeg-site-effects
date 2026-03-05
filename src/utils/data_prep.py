import os
import pandas as pd

PATHOLOGY_LABEL_MAP = {'norm': 0, 'patho': 1, 'normal': 0, 'pathological': 1}

COLUMN_RENAME_MAP = {
    'age_dec': 'age', 'patient_sex': 'gender',
    'institution_id': 'hospital_id', 'classification': 'pathology_label'
}


def load_experiment_data(info_path, features_path):
    """Load info+features CSVs, apply standard column renaming."""
    info_df = pd.read_csv(info_path)
    features_df = pd.read_csv(features_path)
    info_df = info_df.rename(columns=COLUMN_RENAME_MAP)
    return info_df, features_df


def prepare_pathology_labels(info_df):
    """Set hospital_id as categorical, map pathology labels to 0/1.
    Returns y (binary Series).
    """
    all_hospitals = info_df['hospital_id'].unique()
    info_df['hospital_id'] = info_df['hospital_id'].astype(
        pd.CategoricalDtype(categories=all_hospitals, ordered=False)
    )
    return info_df['pathology_label'].map(PATHOLOGY_LABEL_MAP)


def append_results_csv(df, path):
    """Append DataFrame to CSV. Write header only if file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)
