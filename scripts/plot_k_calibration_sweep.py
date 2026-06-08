#!/usr/bin/env python
"""
k_calibration sweep figure.

Reads results/tables/03_harmonization/patho_clf_results.csv and aggregates by
(model, method, n_calib) → mean LOSO MCC and AUC across the 30 hospitals.
The §A baseline rows (no tag, n_calib=30) provide the k=30 point; the
explicitly tagged k5/k15/k50 rows provide the rest of the curve.

Output:
  results/figures/03_harmonization/k_calibration_sweep.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/tables/03_harmonization/patho_clf_results.csv"
OUT_DIR = REPO / "results/figures/03_harmonization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["raw", "sitewise", "combat", "neurocombat", "covbat"]
COLOR_MAP = {"raw": "black", "sitewise": "tomato", "combat": "skyblue",
             "neurocombat": "green", "covbat": "orange"}


def main():
    df = pd.read_csv(RESULTS)
    df = df[df["model"] == "catboost"]
    # Per-fold values; aggregate to (method, n_calib) means
    grouped = df.groupby(["method", "n_calib"]).agg(
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        n_folds=("mcc", "count"),
    ).reset_index()

    print("Aggregated k-sweep:")
    print(grouped.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric in zip(axes, ["mcc", "auc"]):
        for method in METHODS:
            m = grouped[grouped["method"] == method].sort_values("n_calib")
            if m.empty:
                continue
            ax.errorbar(m["n_calib"], m[f"{metric}_mean"], yerr=m[f"{metric}_std"],
                        marker="o", linestyle="-", capsize=3,
                        color=COLOR_MAP[method], label=method, alpha=0.85, linewidth=1.5)
        ax.set_xlabel("Calibration sample size $k$ (normals from held-out site)", fontsize=12)
        ax.set_ylabel(f"LOSO pathology {metric.upper()}", fontsize=12)
        ax.set_xticks(sorted(grouped["n_calib"].unique()))
        ax.grid(axis="both", linestyle="--", alpha=0.4)

    axes[0].set_title("Pathology MCC vs calibration sample size", fontsize=13)
    axes[1].set_title("Pathology AUC vs calibration sample size", fontsize=13)
    axes[1].legend(loc="lower right", fontsize=10, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"k_calibration_sweep.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_DIR}/k_calibration_sweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
