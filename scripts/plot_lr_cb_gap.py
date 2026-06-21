#!/usr/bin/env python
"""
Grouped paired bar chart: LR vs CatBoost site MCC across all
harmonisation conditions. Two adjacent bars per condition (LR / CB)
with SD whiskers and the ΔMCC = CB - LR annotated above each pair.

Conditions are organised into three visually separated blocks:
  Block 1 (no PCA):     Unharmonised, Site-wise, neuroCombat, CovBat
  Block 2 (with PCA):   PCA only, PCA+Site-wise, PCA+neuroCombat, PCA+CovBat
  Block 3 (signal-level on h): MINet, MINet-MTL, MINet-DANN  [TBF]

Conditions within each block are ordered by *increasing* ΔMCC.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import apply_style, MODEL_COLORS, MODEL_LABELS

apply_style()

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/figures/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)


METHOD_DISPLAY = {
    "raw":         "Unharmonised",
    "sitewise":    "Site-wise",
    "combat":      "ComBat",
    "neurocombat": "neuroCombat",
    "covbat":      "CovBat",
}


def load_blocks():
    """Return list of (block_label, [conditions]) where each condition is
    (display_name, lr_mean, lr_sd, cb_mean, cb_sd)."""
    # Block 1: no PCA (from §A site_clf_summary)
    site = pd.read_csv(REPO / "results/tables/02_site_effect/site_clf_summary.csv")
    def get_no_pca(method, model):
        r = site[(site.method == method) & (site.model == model)].iloc[0]
        return float(r["MCC_mean"]), float(r["MCC_std"])

    block1 = []
    for method in ["raw", "sitewise", "neurocombat", "covbat"]:
        lr_m, lr_s = get_no_pca(method, "logreg")
        cb_m, cb_s = get_no_pca(method, "catboost")
        block1.append((METHOD_DISPLAY[method], lr_m, lr_s, cb_m, cb_s))

    # Block 2: with PCA (pca_var=all)
    def get_pca(method, model):
        df = pd.read_csv(REPO / f"results/tables/05_pca_sensitivity/pca_sensitivity_results_site_{model}.csv")
        sub = df[(df["pca_var"].astype(str) == "all") & (df["method"] == method)]
        return float(sub["MCC_Overall"].mean()), float(sub["MCC_Overall"].std())

    block2 = []
    pca_methods = [("raw", "PCA only"),
                   ("sitewise", "PCA + Site-wise"),
                   ("neurocombat", "PCA + neuroCombat"),
                   ("covbat", "PCA + CovBat")]
    for method, label in pca_methods:
        lr_m, lr_s = get_pca(method, "logreg")
        cb_m, cb_s = get_pca(method, "catboost")
        block2.append((label, lr_m, lr_s, cb_m, cb_s))

    # Block 3: signal-level on h. Baseline, MTL and DANN are all shown at BOTH
    # backbone depths (1-layer and 2-layer) so the effect of the second layer is
    # directly visible across every training variant.
    dann_df = pd.read_csv(REPO / "results/tables/04_dann/site_clf_results.csv")
    dann_df = dann_df[dann_df["method"] == "raw"]

    def get_dann(tag, model):
        sub = dann_df[(dann_df["tag"] == tag) & (dann_df["model"] == model)]
        return float(sub["mcc_overall"].mean()), float(sub["mcc_overall"].std())

    # Split the signal-level conditions into two depth groups (all 1-layer
    # heads, then all 2-layer heads) so head depth is the primary visual axis,
    # consistent with the DANN-vs-manual figures.
    block3a, block3b = [], []
    for tag, label, dst in [("baseline", "MINet baseline", block3a),
                            ("dann",     "MINet-DANN",     block3a),
                            ("mtl",      "MINet-MTL",      block3a),
                            ("baseline_2layer", "MINet baseline", block3b),
                            ("dann_2layer",     "MINet-DANN",     block3b),
                            ("mtl_2layer",      "MINet-MTL",      block3b)]:
        lr_m, lr_s = get_dann(tag, "logreg")
        cb_m, cb_s = get_dann(tag, "catboost")
        dst.append((label, lr_m, lr_s, cb_m, cb_s))

    return [
        ("Feature-level, without PCA", block1),
        ("Feature-level, with PCA",    block2),
        ("Signal-level, 1L head$^\\dagger$", block3a),
        ("Signal-level, 2L head$^\\dagger$", block3b),
    ]


def plot():
    # Canonical method order (unharmonised, site-wise, neuroCombat, CovBat)
    # is enforced via the insertion order in load_blocks(); no resorting here.
    blocks = load_blocks()

    fig, ax = plt.subplots(figsize=(15, 6.5))

    bar_width = 0.42
    intra_block_gap = 1.0     # spacing between condition pairs within a block
    inter_block_gap = 0.6     # extra gap between blocks (on top of intra-block)

    xticks, xlabels = [], []
    x_cursor = 0.0
    block_spans = []

    for b_idx, (block_label, conds) in enumerate(blocks):
        block_start = x_cursor
        for cond_label, lr_m, lr_s, cb_m, cb_s in conds:
            x = x_cursor
            # LR bar (left)
            ax.bar(x - bar_width/2, lr_m, bar_width,
                   yerr=lr_s, color=MODEL_COLORS["logreg"],
                   alpha=0.9, capsize=3, edgecolor="black", linewidth=0.5)
            # CB bar (right)
            ax.bar(x + bar_width/2, cb_m, bar_width,
                   yerr=cb_s, color=MODEL_COLORS["catboost"],
                   alpha=0.9, capsize=3, edgecolor="black", linewidth=0.5)
            # ΔMCC annotation above the pair
            gap = cb_m - lr_m
            top = max(lr_m + lr_s, cb_m + cb_s) + 0.04
            ax.annotate(f"Δ={gap:+.2f}", xy=(x, top),
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color="#222222")
            xticks.append(x)
            xlabels.append(cond_label)
            x_cursor += intra_block_gap
        block_end = x_cursor - intra_block_gap
        block_spans.append((block_start, block_end, block_label))
        x_cursor += inter_block_gap

    # Block headers (centred above each block, bold). Sit just above
    # the highest ΔMCC annotation, not at the top of the figure.
    for a, b, label in block_spans:
        ax.text((a + b) / 2, 1.04, label, ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right",
                       fontsize=12, fontweight="bold")
    ax.set_ylabel("Site classification MCC", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Tighten horizontal margins so bars fill the panel
    ax.set_xlim(-bar_width - 0.3, x_cursor - inter_block_gap + bar_width + 0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=MODEL_COLORS["logreg"],   label=MODEL_LABELS["logreg"]),
        Patch(color=MODEL_COLORS["catboost"], label=MODEL_LABELS["catboost"]),
    ]
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.0, 0.98), fontsize=12, framealpha=0.9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"lr_cb_gap.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved {out.relative_to(REPO)}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
