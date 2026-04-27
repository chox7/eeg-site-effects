#!/usr/bin/env python
"""
Relabel hospital IDs in data and results to match the size-based mapping in
data/ELM19/hospital_anonymization_mapping.csv.

Background: ELM19 data files currently use an arbitrary H_i mapping (likely
patho-score based) while the anonymization mapping CSV defines a size-based
ordering (SZC=H1 with 10,482 samples, ..., SL2=H30 with 170 samples). This
script aligns the data to the mapping CSV.

Discovery is pure count-matching:
  - Group current ELM19_info_filtered.csv by H_i  → 30 unique sample counts.
  - Group ELM19_enriched_info_filtered.csv by original code → 27 unique counts.
  - Match by count: current_H ↔ original_code (counts are all distinct).
  - For 3 hospitals missing from enriched (PIO, CHE, SL2), use the mapping
    CSV's size-sorted slot — its ordering matches the unaccounted-for counts.

Usage:
    python scripts/relabel_hospitals.py                 # dry run (default)
    python scripts/relabel_hospitals.py --apply         # actually rewrite files
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATA_INFO_HID = REPO / "data/ELM19/filtered/ELM19_info_filtered.csv"
DATA_INFO_ORIG = REPO / "data/ELM19/filtered/ELM19_enriched_info_filtered.csv"
MAPPING_CSV = REPO / "data/ELM19/hospital_anonymization_mapping.csv"

# Files whose institution_id column needs rewriting
DATA_CSV_GLOBS = [
    "data/ELM19/filtered/ELM19_info_filtered*.csv",
    "data/ELM19/dann/newlabels/ELM19_info_dann*.csv",
]
DATA_INSTITUTION_COL = "institution_id"

# Files already processed in a prior run — skip on subsequent runs to avoid
# double-renaming. Listed as repo-relative paths (matched against
# Path.relative_to(REPO).as_posix()).
ALREADY_DONE: set[str] = {
    # First partial run
    "data/ELM19/dann/newlabels/ELM19_info_dann_newlabels.csv",
    "data/ELM19/dann/newlabels/ELM19_info_dann_norm_newlabels.csv",
    "data/ELM19/filtered/ELM19_info_filtered.csv",
    "data/ELM19/filtered/ELM19_info_filtered_newlabels.csv",
    "data/ELM19/filtered/ELM19_info_filtered_norm.csv",
    # Second partial run — data files succeeded before pipelines crashed
    "data/ELM19/filtered/ELM19_info_filtered_norm_newlabels.csv",
    "data/ELM19/filtered/ELM19_info_filtered_oldlabels.csv",
    # Results CSVs were successfully rewritten in second partial run
    "results/tables/newlabels/dann_patho_clf_results.csv",
    "results/tables/newlabels/dann_site_clf_results.csv",
    "results/tables/newlabels/patho_clf_results.csv",
    "results/tables/newlabels/site_clf_results.csv",
}

# Results CSVs: rewrite `hospital` column values + rename mcc_H* columns
RESULTS_CSV_GLOBS = ["results/tables/newlabels/*.csv"]

# Pipeline files: filename has H_i, e.g. "raw_H10_pipeline.joblib"
PIPELINE_GLOBS = ["models/newlabels/**/*.joblib"]


def parse_mapping_csv() -> list[tuple[str, str]]:
    """Parse the mapping CSV, repairing the corrupted SRK,H24/LUMICE,H25 row."""
    rows: list[tuple[str, str]] = []
    with open(MAPPING_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Original"):
                continue
            if line.count(",") > 1:
                # Known corruption: 'SRK,H2qLUMICE,H25' → SRK,H24 + LUMICE,H25
                if "SRK" in line and "LUMICE" in line:
                    rows.append(("SRK", "H24"))
                    rows.append(("LUMICE", "H25"))
                    continue
                raise ValueError(f"Unexpected multi-comma line: {line!r}")
            orig, hid = line.split(",")
            rows.append((orig, hid))
    if len(rows) != 30:
        raise ValueError(f"Mapping CSV should have 30 rows, got {len(rows)}")
    return rows


# Rename map derived from the first (auto) dry-run, when data files were still
# in the OLD patho-score-based scheme. Frozen here so subsequent runs don't
# collapse to identity now that data files are partially remapped.
HARDCODED_RENAME: dict[str, str] = {
    "H1": "H25", "H2": "H17", "H3": "H16", "H4": "H23", "H5": "H26",
    "H6": "H10", "H7": "H19", "H8": "H11", "H9": "H24", "H10": "H1",
    "H11": "H27", "H12": "H14", "H13": "H29", "H14": "H4", "H15": "H7",
    "H16": "H3", "H17": "H12", "H18": "H9", "H19": "H28", "H20": "H21",
    "H21": "H30", "H22": "H5", "H23": "H20", "H24": "H6", "H25": "H2",
    "H26": "H13", "H27": "H8", "H28": "H22", "H29": "H18", "H30": "H15",
}

# original_code is reconstructed from the mapping CSV (size-sorted by intent).
HARDCODED_AUDIT_ORIGINAL: dict[str, str] = {
    "H25": "LUMICE", "H17": "OST", "H16": "ZOZLO", "H23": "MKW", "H26": "CMD",
    "H10": "LUX_A", "H19": "TOR", "H11": "OTW", "H24": "SRK", "H1": "SZC",
    "H27": "PUS", "H14": "AKS", "H29": "KIEG", "H4": "B2K", "H7": "ARCHDAM",
    "H3": "KATMOJPRZ", "H12": "PRZ", "H9": "SLU", "H28": "MOR", "H21": "KUD",
    "H30": "SL2", "H5": "Z04O", "H20": "CHE", "H6": "WLU", "H2": "STG1",
    "H13": "KLU", "H8": "GAK", "H22": "KAL", "H18": "TER_L", "H15": "PIO",
}


def build_rename_map() -> tuple[dict[str, str], pd.DataFrame]:
    """Return the frozen current_H → new_H map plus an audit DataFrame."""
    rows = []
    for cur_H, new_H in HARDCODED_RENAME.items():
        rows.append({"current_H": cur_H, "new_H": new_H, "original_code": HARDCODED_AUDIT_ORIGINAL[new_H]})
    audit = pd.DataFrame(rows)
    audit["current_H_num"] = audit["current_H"].str[1:].astype(int)
    audit = audit.sort_values("current_H_num").drop(columns=["current_H_num"]).reset_index(drop=True)
    return dict(HARDCODED_RENAME), audit


# ---------- file rewriters ----------

def collect_files() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for tag, globs in (("data", DATA_CSV_GLOBS), ("results", RESULTS_CSV_GLOBS), ("pipelines", PIPELINE_GLOBS)):
        files: list[Path] = []
        for g in globs:
            files.extend(Path(p) for p in glob.glob(str(REPO / g), recursive=True))
        # Drop already-processed paths
        files = [p for p in files if p.relative_to(REPO).as_posix() not in ALREADY_DONE]
        out[tag] = sorted(files)
    return out


def rewrite_data_csv(path: Path, rename: dict[str, str], apply: bool) -> int:
    df = pd.read_csv(path)
    if DATA_INSTITUTION_COL not in df.columns:
        return 0
    before = df[DATA_INSTITUTION_COL].value_counts()
    new_col = df[DATA_INSTITUTION_COL].map(rename)
    if new_col.isna().any():
        unknowns = df[DATA_INSTITUTION_COL][new_col.isna()].unique()
        raise ValueError(f"{path.name}: unknown H_i values not in rename map: {unknowns}")
    n_changed = (new_col.values != df[DATA_INSTITUTION_COL].values).sum()
    if apply:
        df[DATA_INSTITUTION_COL] = new_col
        df.to_csv(path, index=False)
    return int(n_changed)


def rewrite_results_csv(path: Path, rename: dict[str, str], apply: bool) -> tuple[int, int]:
    df = pd.read_csv(path)
    n_row_changed = 0
    n_col_changed = 0

    if "hospital" in df.columns:
        new_col = df["hospital"].map(rename).fillna(df["hospital"])
        n_row_changed = (new_col.values != df["hospital"].values).sum()
        if apply:
            df["hospital"] = new_col

    mcc_pat = re.compile(r"^mcc_H(\d+)$", re.IGNORECASE)
    col_renames: dict[str, str] = {}
    for c in df.columns:
        m = mcc_pat.match(c)
        if not m:
            continue
        old_h = f"H{m.group(1)}"
        if old_h in rename:
            new_c = c[: m.start(1) - 1] + rename[old_h]  # preserve "mcc_" or "MCC_" prefix casing-wise
            # Simpler: rebuild
            prefix = c[:4]  # "mcc_" or "MCC_"
            new_c = f"{prefix}{rename[old_h]}"
            col_renames[c] = new_c
    n_col_changed = sum(1 for k, v in col_renames.items() if k != v)
    if apply and col_renames:
        df = df.rename(columns=col_renames)

    if apply and (n_row_changed or n_col_changed):
        df.to_csv(path, index=False)
    return int(n_row_changed), int(n_col_changed)


_HI_PAT = re.compile(r"(?P<lead>^|_)(?P<h>H\d+)(?P<trail>_|$)")


def _planned_target(path: Path, rename: dict[str, str]) -> Path | None:
    """Compute the eventual renamed path for a pipeline file (or None if no rename)."""
    m = _HI_PAT.search(path.name)
    if not m:
        return None
    old_h = m.group("h")
    if old_h not in rename or rename[old_h] == old_h:
        return None
    new_name = path.name[: m.start("h")] + rename[old_h] + path.name[m.end("h") :]
    return path.with_name(new_name)


def rename_pipelines_two_pass(paths: list[Path], rename: dict[str, str], apply: bool) -> int:
    """Two-pass rename to avoid collisions when source/target H_i swap among files.

    Pass 1: every file gets a unique '.RELABEL_TMP' suffix on its target name.
    Pass 2: strip the suffix to land on the final name.
    """
    pairs = []  # (orig_path, final_path)
    for p in paths:
        target = _planned_target(p, rename)
        if target is None or target == p:
            continue
        pairs.append((p, target))

    if not apply:
        return len(pairs)

    # Pass 1: orig → final + ".RELABEL_TMP"
    tmp_pairs = []  # (tmp_path, final_path)
    for orig, final in pairs:
        tmp = final.with_name(final.name + ".RELABEL_TMP")
        if tmp.exists():
            raise FileExistsError(f"Stale tmp file: {tmp}")
        orig.rename(tmp)
        tmp_pairs.append((tmp, final))

    # Pass 2: tmp → final. By construction, no two `final` paths collide
    # (rename map is a bijection on the 30 H_i values).
    for tmp, final in tmp_pairs:
        if final.exists():
            raise FileExistsError(f"Final target collision: {final}")
        tmp.rename(final)

    return len(pairs)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually rewrite files (default is dry run)")
    args = ap.parse_args()

    rename, audit = build_rename_map()
    no_change = sum(1 for k, v in rename.items() if k == v)
    will_change = 30 - no_change

    print("=== Hospital ID rename plan ===")
    print(f"Hospitals: 30  |  unchanged: {no_change}  |  to rename: {will_change}")
    print()
    cols = [c for c in ["current_H", "new_H", "original_code", "size"] if c in audit.columns]
    print(audit[cols].to_string(index=False))

    files = collect_files()
    print("\n=== Files affected ===")
    for tag, paths in files.items():
        print(f"  {tag}: {len(paths)} file(s)")
        for p in paths[:5]:
            print(f"    {p.relative_to(REPO)}")
        if len(paths) > 5:
            print(f"    ... and {len(paths) - 5} more")

    if not args.apply:
        print("\n(dry run — pass --apply to perform the rewrites)")
        return

    print("\n=== Applying rewrites ===")
    # Backup mapping CSV (only the corrupted line is being fixed)
    backup = MAPPING_CSV.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy2(MAPPING_CSV, backup)
        print(f"  backed up mapping → {backup.relative_to(REPO)}")

    # Rewrite mapping CSV cleanly
    mapping_rows = parse_mapping_csv()
    with open(MAPPING_CSV, "w") as f:
        f.write("Original_Hospital,Anonymized_ID\n")
        for orig, hid in mapping_rows:
            f.write(f"{orig},{hid}\n")
    print(f"  wrote clean mapping → {MAPPING_CSV.relative_to(REPO)}")

    total_data_changed = 0
    for p in files["data"]:
        n = rewrite_data_csv(p, rename, apply=True)
        total_data_changed += n
        print(f"  data: {p.relative_to(REPO)} — {n} row values updated")
    print(f"  data subtotal: {total_data_changed} row values rewritten")

    total_row = total_col = 0
    for p in files["results"]:
        nr, nc = rewrite_results_csv(p, rename, apply=True)
        total_row += nr
        total_col += nc
        print(f"  results: {p.relative_to(REPO)} — {nr} hospital cells, {nc} columns renamed")
    print(f"  results subtotal: {total_row} cells, {total_col} column renames")

    n_pipe = rename_pipelines_two_pass(files["pipelines"], rename, apply=True)
    print(f"  pipelines: {n_pipe} files renamed (two-pass)")

    print("\nDone. Sanity check the new H_i ordering before committing.")


if __name__ == "__main__":
    main()
