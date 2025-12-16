#!/usr/bin/env python
"""
Generate info.csv files for each filter experiment by matching
metrics.csv (which contains successful exam_ids) with the original info file.
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

        # Load metrics to get successful exam_ids
        df_metrics = pd.read_csv(metrics_path)
        successful_exam_ids = df_metrics['examination_id'].unique()

        # Filter info to only successful exams
        df_info_filtered = df_info[df_info['examination_id'].isin(successful_exam_ids)].reset_index(drop=True)

        # Verify count matches features
        if features_path.exists():
            n_features = len(pd.read_csv(features_path))
            if len(df_info_filtered) != n_features:
                print(f"[WARN] {config_name}: info has {len(df_info_filtered)} rows but features has {n_features}")

        # Save
        df_info_filtered.to_csv(info_path, index=False)
        print(f"[OK] {config_name}: saved {len(df_info_filtered)} records to info.csv")

if __name__ == "__main__":
    main()
