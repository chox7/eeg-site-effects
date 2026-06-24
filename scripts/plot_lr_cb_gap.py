#!/usr/bin/env python
"""
The LR-vs-CatBoost site-MCC figure (the "harmonisation paradox" fingerprint).

Two probes are trained on the SAME representation for every condition:
  - LR  = linear probe             (orange)
  - CB  = CatBoost nonlinear probe (blue)
A large LR<<CB gap means residual site information is hidden in nonlinear
covariance structure (marginal correction reorganises rather than removes it);
LR~CB means the residual site structure is genuinely reduced (adversarial).

Conditions are organised into four visually separated blocks:
  Block 1  Feature-level, without PCA   (Unharmonised, Site-wise, neuroCombat, CovBat)
  Block 2  Feature-level, with PCA      (PCA only, +Site-wise, +neuroCombat, +CovBat)
  Block 3  Signal-level on h, 1L head   (MINet-1L, MINet-DANN-1L, MINet-MTL-1L)
  Block 4  Signal-level on h, 2L head   (MINet-2L, MINet-DANN-2L, MINet-MTL-2L)

Rendered in the project's paired house style (house probe colours, translucent
fill, bold black edges). Two outputs:
  lr_cb_gap.{png,pdf}              PRIMARY — grouped bars of the *overall* (pooled,
                                   multiclass) site MCC, mean +/- SD across folds.
                                   These are the SAME numbers as Table
                                   tab:site_class_harmonisation, so figure and
                                   table are consistent.
  lr_cb_gap_perhospital.{png,pdf}  ALT — paired box distributions of the *per-
                                   hospital* one-vs-rest MCC (30 sites). A DIFFERENT
                                   statistic (mean-per-hospital != pooled overall),
                                   shown for the site-level spread; not table-matching.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.visualization.style import apply_style, MODEL_COLORS, MODEL_LABELS

apply_style()

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/figures/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LR, CB = MODEL_COLORS["logreg"], MODEL_COLORS["catboost"]

# ---- data sources ----------------------------------------------------------
D2 = pd.read_csv(REPO / "results/tables/02_site_effect/site_clf_results.csv")
D4 = pd.read_csv(REPO / "results/tables/04_dann/site_clf_results.csv")
D5 = {m: pd.read_csv(REPO / f"results/tables/05_pca_sensitivity/pca_sensitivity_results_site_{m}.csv")
      for m in ["logreg", "catboost"]}


def _hcols(df):
    return [c for c in df.columns if re.match(r"(?i)mcc_H\d+$", c)]


def _subset(source, model, key):
    """Return the rows for one (condition, model)."""
    if source == "02":
        return D2[(D2.model == model) & (D2.method == key)]
    if source == "05":
        d = D5[model]
        return d[(d.pca_var.astype(str) == "all") & (d.method == key)]
    if source == "04":
        return D4[(D4.model == model) & (D4.method == "raw") & (D4.tag == key)]
    raise ValueError(source)


def _overall_col(source):
    return "MCC_Overall" if source == "05" else "mcc_overall"


def overall(source, model, key):
    """Pooled multiclass site MCC: (mean, std) across folds — matches the table."""
    s = _subset(source, model, key)[_overall_col(source)]
    return float(s.mean()), float(s.std())


def perhosp(source, model, key):
    """Per-hospital one-vs-rest MCC as a Series indexed by hospital id."""
    sub = _subset(source, model, key)
    cols = _hcols(sub)
    s = sub[cols].mean()
    s.index = [re.sub(r"(?i)^mcc_", "", c) for c in cols]
    return s


# (block label, [(condition label, source, key), ...])
CONDS = [
    ("Feature-level,\nwithout PCA", [
        ("Unharmonised", "02", "raw"),
        ("Site-wise",    "02", "sitewise"),
        ("neuroCombat",  "02", "neurocombat"),
        ("CovBat",       "02", "covbat")]),
    ("Feature-level,\nwith PCA", [
        ("PCA only",          "05", "raw"),
        ("PCA + Site-wise",   "05", "sitewise"),
        ("PCA + neuroCombat", "05", "neurocombat"),
        ("PCA + CovBat",      "05", "covbat")]),
    ("Signal-level,\n1L head", [
        ("MINet-1L",      "04", "baseline"),
        ("MINet-DANN-1L", "04", "dann"),
        ("MINet-MTL-1L",  "04", "mtl")]),
    ("Signal-level,\n2L head", [
        ("MINet-2L",      "04", "baseline_2layer"),
        ("MINet-DANN-2L", "04", "dann_2layer"),
        ("MINet-MTL-2L",  "04", "mtl_2layer")]),
]

_LEGEND = [Patch(facecolor=LR, alpha=0.55, edgecolor="black",
                 label=MODEL_LABELS["logreg"] + " (linear probe)"),
           Patch(facecolor=CB, alpha=0.55, edgecolor="black",
                 label=MODEL_LABELS["catboost"] + " (nonlinear probe)")]


def _save(fig, stem):
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved {out.relative_to(REPO)}")
    plt.close(fig)


def _block_headers(ax, spans, y):
    for i, (a, b, lbl) in enumerate(spans):
        ax.axvspan(a - 0.55, b + 0.55, color="#f6f6f6" if i % 2 == 0 else "#ffffff", zorder=0)
        ax.text((a + b) / 2, y, lbl, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")


# ---- PRIMARY: grouped bars on overall MCC (matches the paper table) --------
def plot_bars():
    bw = 0.40
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = 0.0
    centers, labels, spans = [], [], []
    for blabel, conds in CONDS:
        start = x
        for cname, src, key in conds:
            lr_m, lr_s = overall(src, "logreg", key)
            cb_m, cb_s = overall(src, "catboost", key)
            ax.bar(x - bw / 2, lr_m, bw, yerr=lr_s, facecolor=LR, alpha=0.55,
                   edgecolor="black", linewidth=1.4, capsize=3, zorder=2,
                   error_kw=dict(ecolor="black", lw=1.1))
            ax.bar(x + bw / 2, cb_m, bw, yerr=cb_s, facecolor=CB, alpha=0.55,
                   edgecolor="black", linewidth=1.4, capsize=3, zorder=2,
                   error_kw=dict(ecolor="black", lw=1.1))
            top = max(lr_m + lr_s, cb_m + cb_s) + 0.03
            ax.annotate(f"$\\Delta${cb_m - lr_m:+.2f}", (x, top), ha="center",
                        va="bottom", fontsize=10.5, fontweight="bold", color="#222")
            centers.append(x); labels.append(cname); x += 1.0
        spans.append((start, x - 1.0, blabel)); x += 0.85
    _block_headers(ax, spans, y=1.10)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10.5)
    ax.set_xlim(-0.7, x - 0.85 + 0.7)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Site classification MCC", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles=_LEGEND, loc="lower right", bbox_to_anchor=(1.0, 1.03),
              fontsize=10, framealpha=0.95, ncol=2)
    fig.tight_layout()
    _save(fig, "lr_cb_gap")


# ---- ALT: paired per-hospital box distributions (different statistic) ------
def _boxes(ax, positions, data, color, bw):
    bp = ax.boxplot(data, positions=positions, widths=bw, notch=False,
                    showfliers=False, patch_artist=True, zorder=2,
                    medianprops=dict(color="black", linewidth=2.3),
                    whiskerprops=dict(color="black", linewidth=1.3),
                    capprops=dict(color="black", linewidth=1.3),
                    boxprops=dict(edgecolor="black", linewidth=1.3))
    for p in bp["boxes"]:
        p.set_facecolor(color)
        p.set_alpha(0.30)


def plot_perhospital():
    dx, bw = 0.21, 0.34
    fig, ax = plt.subplots(figsize=(15.5, 7))
    x = 0.0
    centers, labels, spans = [], [], []
    lr_pos, lr_data, cb_pos, cb_data = [], [], [], []
    for blabel, conds in CONDS:
        start = x
        for cname, src, key in conds:
            slr, scb = perhosp(src, "logreg", key), perhosp(src, "catboost", key)
            common = [h for h in slr.index if h in scb.index]
            lpos, cpos = x - dx, x + dx
            for h in common:
                ax.plot([lpos, cpos], [slr[h], scb[h]], color="gray",
                        alpha=0.20, lw=0.9, zorder=1)
            lr_pos.append(lpos); lr_data.append(slr.values)
            cb_pos.append(cpos); cb_data.append(scb.values)
            centers.append(x); labels.append(cname); x += 1.0
        spans.append((start, x - 1.0, blabel)); x += 0.9
    _boxes(ax, lr_pos, lr_data, LR, bw)
    _boxes(ax, cb_pos, cb_data, CB, bw)
    for pos, data, color in [(lr_pos, lr_data, LR), (cb_pos, cb_data, CB)]:
        for p, d in zip(pos, data):
            ax.scatter(np.full(len(d), p), d, color=color, s=14, alpha=0.75,
                       edgecolor="black", linewidth=0.4, zorder=3)
    _block_headers(ax, spans, y=1.08)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10.5)
    ax.set_xlim(-0.7, x - 0.9 + 0.7)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Per-hospital site MCC", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles=_LEGEND, loc="lower right", bbox_to_anchor=(1.0, 1.03),
              fontsize=10, framealpha=0.95, ncol=2)
    fig.tight_layout()
    _save(fig, "lr_cb_gap_perhospital")


if __name__ == "__main__":
    plot_bars()
    plot_perhospital()
