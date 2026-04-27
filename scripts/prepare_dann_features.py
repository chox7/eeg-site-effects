#!/usr/bin/env python
"""
Convert DANN-extracted features (.npz) into CSV matrices aligned with the
hybrid-label info file, ready for pathology_classification.py.

Produces one info file (intersection with ELM19_info_filtered_newlabels.csv)
and one features CSV per DANN variant (baseline / mtl / dann). All outputs
share the same row ordering so any variant can be swapped in via config.
"""
import numpy as np
import pandas as pd


NPZ_DIR = "data/ELM19/dann/newlabels"
INFO_PATH = "data/ELM19/filtered/ELM19_info_filtered_newlabels.csv"
OUT_DIR = "data/ELM19/dann/newlabels"

VARIANTS = {
    "baseline": "extracted_features_baseline_final.npz",
    "mtl": "extracted_features_mtl_final.npz",
    "dann": "extracted_features_dann_final.npz",
}


UUID_PATTERN = r"(\{[0-9a-fA-F-]+\})"


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        "uuid": pd.Series(d["eids"].astype(str)).str.extract(UUID_PATTERN, expand=False),
        **{f"f{i:03d}": d["features"][:, i] for i in range(d["features"].shape[1])},
    })


def main():
    print(f"Loading info: {INFO_PATH}")
    info = pd.read_csv(INFO_PATH)
    info["examination_id"] = info["examination_id"].astype(str)
    # Normalize: info has '<date>-<time>-{uuid}', DANN npz has '{uuid}' only.
    info["uuid"] = info["examination_id"].str.extract(UUID_PATTERN, expand=False)
    assert info["uuid"].notna().all(), "some info examination_ids have no {uuid} tail"
    print(f"  info rows: {len(info)}")

    # Load all variants, verify they share the same eid set
    variant_dfs = {}
    for name, fname in VARIANTS.items():
        path = f"{NPZ_DIR}/{fname}"
        df = load_npz(path)
        print(f"  {name}: {len(df)} rows, {df.shape[1] - 1} feature dims")
        variant_dfs[name] = df

    uuid_sets = [set(df["uuid"]) for df in variant_dfs.values()]
    assert all(s == uuid_sets[0] for s in uuid_sets), "DANN variants disagree on uuids"

    # Intersection with info, preserve info's row order
    info_uuid_set = set(info["uuid"])
    dann_uuid_set = uuid_sets[0]
    common = info_uuid_set & dann_uuid_set
    print(f"\n  info ∩ dann: {len(common)} samples "
          f"(info-only: {len(info_uuid_set - dann_uuid_set)}, dann-only: {len(dann_uuid_set - info_uuid_set)})")

    info_aligned = info[info["uuid"].isin(common)].reset_index(drop=True)
    info_out = f"{OUT_DIR}/ELM19_info_dann_newlabels.csv"
    info_aligned.drop(columns=["uuid"]).to_csv(info_out, index=False)
    print(f"\n  Saved aligned info: {info_out} ({len(info_aligned)} rows)")

    for name, df in variant_dfs.items():
        df_aligned = df.set_index("uuid").loc[info_aligned["uuid"]].reset_index(drop=True)
        feat_out = f"{OUT_DIR}/ELM19_features_dann_{name}_newlabels.csv"
        df_aligned.to_csv(feat_out, index=False)
        print(f"  Saved {name}: {feat_out} ({df_aligned.shape})")

    # Normals-only subset for site-classification probe
    norm_mask = (info_aligned["classification"] == "norm").values
    norm_info = info_aligned[norm_mask].drop(columns=["uuid"]).reset_index(drop=True)
    norm_info_out = f"{OUT_DIR}/ELM19_info_dann_norm_newlabels.csv"
    norm_info.to_csv(norm_info_out, index=False)
    print(f"\n  Saved normals-only info: {norm_info_out} ({len(norm_info)} rows)")

    for name, df in variant_dfs.items():
        df_aligned = df.set_index("uuid").loc[info_aligned["uuid"]].reset_index(drop=True)
        norm_feat = df_aligned[norm_mask].reset_index(drop=True)
        norm_feat_out = f"{OUT_DIR}/ELM19_features_dann_{name}_norm_newlabels.csv"
        norm_feat.to_csv(norm_feat_out, index=False)
        print(f"  Saved {name} normals: {norm_feat_out} ({norm_feat.shape})")


if __name__ == "__main__":
    main()
