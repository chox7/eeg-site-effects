"""Anonymize hospital IDs in info CSV files using the mapping file."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Anonymize hospital IDs in info CSVs")
    parser.add_argument("--mapping", default="data/ELM19/hospital_anonymization_mapping.csv",
                        help="Path to anonymization mapping CSV")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Info CSV files to anonymize (modified in-place)")
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping).dropna()
    anon_map = dict(zip(mapping["Original_Hospital"], mapping["Anonymized_ID"]))
    print(f"Loaded mapping: {len(anon_map)} hospitals")

    for fpath in args.files:
        df = pd.read_csv(fpath)
        col = "institution_id"
        if col not in df.columns:
            print(f"SKIP {fpath}: no '{col}' column")
            continue

        before = df[col].unique()
        df[col] = df[col].map(anon_map)
        unmapped = df[col].isna().sum()
        if unmapped > 0:
            print(f"WARNING: {unmapped} rows have unmapped institution_id in {fpath}")

        df.to_csv(fpath, index=False)
        print(f"Anonymized {fpath}: {len(before)} hospitals → {df[col].nunique()} anonymized IDs")


if __name__ == "__main__":
    main()
