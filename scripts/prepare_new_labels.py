#!/usr/bin/env python
"""
Prepare ELM19 data with hybrid new labels (CLAS_SVC_1B).

For every sample in the filtered ELM19 dataset, use the new CLAS_SVC_1B label
when available; fall back to the original `classification` value otherwise.
This preserves the full sample count while migrating to the new labelling.

Outputs:
  - data/ELM19/filtered/ELM19_info_filtered_newlabels.csv
  - data/ELM19/filtered/ELM19_features_filtered_newlabels.csv
"""

import pandas as pd


INFO_PATH = "data/ELM19/filtered/ELM19_info_filtered.csv"
FEATURES_PATH = "data/ELM19/filtered/ELM19_features_filtered.csv"
NEW_LABELS_PATH = "data/base_92_094_pred_NEW2.csv"
OUT_DIR = "data/ELM19/filtered"

NEW_LABEL_MAP = {"NORM": "norm", "PAT": "patho"}


def main():
    print("Loading data...")
    info = pd.read_csv(INFO_PATH)
    features = pd.read_csv(FEATURES_PATH)
    new_labels = pd.read_csv(
        NEW_LABELS_PATH, sep="|", low_memory=False,
        usecols=["examination_id", "CLAS_SVC_1B"],
    )

    assert len(info) == len(features), "info/features row count mismatch"
    n_total = len(info)
    print(f"  Info/features: {n_total} rows")
    print(f"  New labels file: {len(new_labels)} rows")

    # Left-merge to preserve every original row
    merged = info.merge(new_labels, on="examination_id", how="left")
    assert len(merged) == n_total, "left merge changed row count"

    mapped_new = merged["CLAS_SVC_1B"].map(NEW_LABEL_MAP)
    n_with_new = mapped_new.notna().sum()
    n_fallback = n_total - n_with_new
    print(f"  Samples with new label: {n_with_new}")
    print(f"  Samples falling back to old label: {n_fallback}")

    hybrid = mapped_new.fillna(merged["classification"])
    n_changed = (hybrid != merged["classification"]).sum()
    print(f"  Labels changed vs. old: {n_changed}")

    out_info = merged.drop(columns=["CLAS_SVC_1B"]).copy()
    out_info["classification"] = hybrid.values

    info_out = f"{OUT_DIR}/ELM19_info_filtered_newlabels.csv"
    feat_out = f"{OUT_DIR}/ELM19_features_filtered_newlabels.csv"
    out_info.to_csv(info_out, index=False)
    features.to_csv(feat_out, index=False)

    print(f"\n  Saved: {info_out} ({len(out_info)} rows)")
    print(f"  Saved: {feat_out} ({len(features)} rows)")
    print(f"  Final label distribution: {out_info['classification'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
