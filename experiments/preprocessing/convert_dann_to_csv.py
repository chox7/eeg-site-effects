"""Convert DANN extracted features from NPZ to CSV format aligned with info file."""

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert DANN NPZ features to CSV")
    parser.add_argument("--npz", required=True, help="Path to extracted_features*.npz")
    parser.add_argument("--info", required=True, help="Path to ELM19 info CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Load NPZ
    data = np.load(args.npz, allow_pickle=True)
    features = data["features"]  # (N, 288)
    eids = data["eids"].astype(str)  # (N,)

    print(f"NPZ: {features.shape[0]} samples, {features.shape[1]} features")

    # Build DataFrame indexed by eid
    col_names = [f"dann_feat_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, index=eids, columns=col_names)
    df.index.name = "eid"

    # Load info CSV to get canonical row order
    info = pd.read_csv(args.info)
    exam_ids = info["examination_id"].values
    print(f"Info: {len(exam_ids)} examination IDs")

    # Filter to matching eids and reorder
    common = set(df.index) & set(exam_ids)
    only_npz = set(df.index) - set(exam_ids)
    only_info = set(exam_ids) - set(df.index)

    print(f"Common: {len(common)}, NPZ-only: {len(only_npz)}, Info-only: {len(only_info)}")

    if only_info:
        raise ValueError(f"{len(only_info)} info IDs missing from NPZ!")

    # Reindex to match info row order, drop extras
    df_aligned = df.loc[exam_ids]

    # Validate
    assert len(df_aligned) == len(exam_ids), f"Row count mismatch: {len(df_aligned)} vs {len(exam_ids)}"
    assert df_aligned.isna().sum().sum() == 0, "NaN values found!"

    # Save without index (matching ELM19_features_filtered.csv format)
    df_aligned.reset_index(drop=True).to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({df_aligned.shape[0]} rows x {df_aligned.shape[1]} cols)")


if __name__ == "__main__":
    main()
