#!/usr/bin/env python
"""
Generate info.csv files for each filter experiment by matching
metrics.csv (successful files only) with the original info file.
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_BASE = Path("data/ELM19/experiments")
ORIGINAL_INFO = Path("data/ELM19/filtered/ELM19_info_filtered.csv")

# Filter configs to process
FILTER_CONFIGS = [
    "original",
    "remove_low_freq_artifacts",
    "remove_high_freq_artifacts",
    "strict_both_ends",
    "clinical_range",
]

def main():
    # Load original info
    df_info = pd.read_csv(ORIGINAL_INFO)
    print(f"Original info: {len(df_info)} records")

    for config_name in FILTER_CONFIGS:
        config_dir = DATA_BASE / config_name
        metrics_path = config_dir / "metrics.csv"
        features_path = config_dir / "features.csv"
        info_path = config_dir / "info.csv"

        if not config_dir.exists():
            print(f"[SKIP] {config_name}: directory not found")
            continue

        if not metrics_path.exists():
            print(f"[SKIP] {config_name}: metrics.csv not found")
            continue

        # Load metrics and filter for successful files only
        df_metrics = pd.read_csv(metrics_path)

        # Filter by preprocessing_successful flag if it exists
        if 'preprocessing_successful' in df_metrics.columns:
            df_metrics = df_metrics[df_metrics['preprocessing_successful'] == True]

        # Also filter by segments_kept > 0 if the column exists
        if 'segmentation.segments_kept' in df_metrics.columns:
            df_metrics = df_metrics[df_metrics['segmentation.segments_kept'] > 0]

        successful_exam_ids = df_metrics['examination_id'].unique()

        # Filter info to only successful exams, preserving original order
        df_info_filtered = df_info[df_info['examination_id'].isin(successful_exam_ids)].reset_index(drop=True)

        # Verify count matches features
        if features_path.exists():
            n_features = len(pd.read_csv(features_path))
            if len(df_info_filtered) != n_features:
                print(f"[WARN] {config_name}: info has {len(df_info_filtered)} rows but features has {n_features}")
                print(f"       Failed exam_ids might be: check metrics for preprocessing_successful=False")
            else:
                print(f"[OK] {config_name}: {len(df_info_filtered)} records match features")

        # Save
        df_info_filtered.to_csv(info_path, index=False)

if __name__ == "__main__":
    main()
