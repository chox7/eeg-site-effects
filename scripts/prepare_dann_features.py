#!/usr/bin/env python
"""
Convert DANN-extracted features (.npz) into CSV matrices aligned with the
hybrid-label info file, ready for pathology_classification.py.

Produces one info file (intersection with ELM19_info_filtered_newlabels.csv)
and one features CSV per DANN variant. All outputs share the same row
ordering so any variant can be swapped in via config.

Default behaviour reproduces the original 1-layer run (baseline + mtl + dann
from data/ELM19/dann/newlabels/). Use --src-dir / --suffix / --variants to
prepare alternate extractions (e.g. 2-layer) without overwriting old files.
"""
import argparse
import numpy as np
import pandas as pd


DEFAULT_NPZ_DIR = "data/ELM19/dann/newlabels"
DEFAULT_INFO = "data/ELM19/filtered/ELM19_info_filtered_newlabels.csv"
DEFAULT_OUT_DIR = "data/ELM19/dann/newlabels"
DEFAULT_VARIANTS = "baseline,mtl,dann"

UUID_PATTERN = r"(\{[0-9a-fA-F-]+\})"


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        "uuid": pd.Series(d["eids"].astype(str)).str.extract(UUID_PATTERN, expand=False),
        **{f"f{i:03d}": d["features"][:, i] for i in range(d["features"].shape[1])},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", default=DEFAULT_NPZ_DIR,
                        help=f"Directory holding extracted_features_*.npz (default: {DEFAULT_NPZ_DIR})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Directory to write CSVs into (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--info", default=DEFAULT_INFO,
                        help=f"Info CSV to align against (default: {DEFAULT_INFO})")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS,
                        help=f"Comma-separated variant names (default: {DEFAULT_VARIANTS}). "
                             "Each entry is either a bare name (file resolved as "
                             "<src-dir>/extracted_features_<name>_final.npz) or 'name=filename' "
                             "to point a variant at an off-pattern npz, e.g. "
                             "'baseline=extracted_features_baseline_2l_final.npz'. A filename "
                             "containing '/' is treated as a full path, otherwise it is joined to --src-dir.")
    parser.add_argument("--suffix", default="",
                        help="Suffix inserted into output filenames, e.g. '_2layer'. "
                             "info becomes ELM19_info_dann{suffix}_newlabels.csv; "
                             "features become ELM19_features_dann_{variant}{suffix}{_norm}_newlabels.csv. "
                             "Default empty (overwrites the 1-layer outputs).")
    args = parser.parse_args()

    variants = []
    variant_paths = {}
    for tok in args.variants.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, fname = (s.strip() for s in tok.split("=", 1))
            variant_paths[name] = fname if "/" in fname else f"{args.src_dir}/{fname}"
        else:
            name = tok
            variant_paths[name] = f"{args.src_dir}/extracted_features_{name}_final.npz"
        variants.append(name)

    print(f"Loading info: {args.info}")
    info = pd.read_csv(args.info)
    info["examination_id"] = info["examination_id"].astype(str)
    # Normalize: info has '<date>-<time>-{uuid}', DANN npz has '{uuid}' only.
    info["uuid"] = info["examination_id"].str.extract(UUID_PATTERN, expand=False)
    assert info["uuid"].notna().all(), "some info examination_ids have no {uuid} tail"
    print(f"  info rows: {len(info)}")

    variant_dfs = {}
    for name, path in variant_paths.items():
        df = load_npz(path)
        print(f"  {name}: {len(df)} rows, {df.shape[1] - 1} feature dims  ({path})")
        variant_dfs[name] = df

    uuid_sets = [set(df["uuid"]) for df in variant_dfs.values()]
    assert all(s == uuid_sets[0] for s in uuid_sets), "DANN variants disagree on uuids"

    info_uuid_set = set(info["uuid"])
    dann_uuid_set = uuid_sets[0]
    common = info_uuid_set & dann_uuid_set
    print(f"\n  info ∩ dann: {len(common)} samples "
          f"(info-only: {len(info_uuid_set - dann_uuid_set)}, dann-only: {len(dann_uuid_set - info_uuid_set)})")

    info_aligned = info[info["uuid"].isin(common)].reset_index(drop=True)
    info_out = f"{args.out_dir}/ELM19_info_dann{args.suffix}_newlabels.csv"
    info_aligned.drop(columns=["uuid"]).to_csv(info_out, index=False)
    print(f"\n  Saved aligned info: {info_out} ({len(info_aligned)} rows)")

    for name, df in variant_dfs.items():
        df_aligned = df.set_index("uuid").loc[info_aligned["uuid"]].reset_index(drop=True)
        feat_out = f"{args.out_dir}/ELM19_features_dann_{name}{args.suffix}_newlabels.csv"
        df_aligned.to_csv(feat_out, index=False)
        print(f"  Saved {name}: {feat_out} ({df_aligned.shape})")

    # Normals-only subset for site-classification probe
    norm_mask = (info_aligned["classification"] == "norm").values
    norm_info = info_aligned[norm_mask].drop(columns=["uuid"]).reset_index(drop=True)
    norm_info_out = f"{args.out_dir}/ELM19_info_dann{args.suffix}_norm_newlabels.csv"
    norm_info.to_csv(norm_info_out, index=False)
    print(f"\n  Saved normals-only info: {norm_info_out} ({len(norm_info)} rows)")

    for name, df in variant_dfs.items():
        df_aligned = df.set_index("uuid").loc[info_aligned["uuid"]].reset_index(drop=True)
        norm_feat = df_aligned[norm_mask].reset_index(drop=True)
        norm_feat_out = f"{args.out_dir}/ELM19_features_dann_{name}{args.suffix}_norm_newlabels.csv"
        norm_feat.to_csv(norm_feat_out, index=False)
        print(f"  Saved {name} normals: {norm_feat_out} ({norm_feat.shape})")


if __name__ == "__main__":
    main()
