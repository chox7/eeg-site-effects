"""
Shared plot-style module — call ``apply_style()`` from the first cell of any
notebook to get consistent fonts, grid, colours, and figsize defaults
across the entire project.

Usage
-----
    from src.visualization.style import (
        apply_style,
        METHOD_COLORS, MODEL_COLORS, GROUP_COLOR, DANN_TAG_STYLE,
        FIGSIZE_WIDE, FIGSIZE_TALL, FIGSIZE_SQUARE,
        natural_sort_key,
    )
    apply_style()
"""
from __future__ import annotations

import re
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------- rcParams --------------------------------------------------------

_RCPARAMS = {
    # Font
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":        12,
    "axes.titlesize":   16,
    "axes.titleweight": "bold",
    "axes.labelsize":   14,
    "axes.labelweight": "bold",
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 17,
    "figure.titleweight": "bold",
    # Axes / spines / grid — clean journal look
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      1.0,
    "axes.edgecolor":      "#222222",
    "axes.grid":           True,
    "axes.axisbelow":      True,   # grid behind data
    "grid.linestyle":      "--",
    "grid.alpha":          0.4,
    "grid.linewidth":      0.6,
    "grid.color":          "#bbbbbb",
    # Tick marks
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.width":   1.0,
    "ytick.major.width":   1.0,
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    # Legend
    "legend.framealpha":   0.95,
    "legend.edgecolor":    "#444444",
    "legend.fancybox":     False,
    # Figure / save
    "figure.dpi":          110,
    "figure.facecolor":    "white",
    "savefig.dpi":         200,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    # Lines / markers
    "lines.linewidth":     1.8,
    "lines.markersize":    8,
    "lines.markeredgecolor": "#222222",
    "lines.markeredgewidth": 0.6,
}


def apply_style() -> None:
    """Idempotent: set project-wide matplotlib defaults."""
    mpl.rcParams.update(_RCPARAMS)


# ---------- shared colour / style dicts -------------------------------------

#: Harmonisation methods — used in PCA, harmonization, ablation, DANN-vs-manual plots.
METHOD_COLORS: dict[str, str] = {
    "raw":         "#1a1a1a",   # near-black
    "sitewise":    "#e6550d",   # orange
    "combat":      "#4C72B0",   # steel blue
    "neurocombat": "#31a354",   # green
    "covbat":      "#756bb1",   # purple
}

#: PCA variance levels.
PCA_COLORS: dict[str, str] = {
    "none": "#1a1a1a",
    "all":  "#2166ac",
    "0.99": "#4393c3",
    "0.95": "#92c5de",
    "0.9":  "#f4a582",
    "0.8":  "#d6604d",
}

#: Two-classifier framework probes.
MODEL_COLORS: dict[str, str] = {
    "catboost": "#1f77b4",
    "logreg":   "#ff7f0e",
}
MODEL_LABELS: dict[str, str] = {
    "catboost": "CatBoost",
    "logreg":   "Logistic Regression",
}

#: Feature-group ablation: coherence / power / covariance.
GROUP_COLOR: dict[str, str] = {
    "coh": "#d62728",   # red
    "pow": "#1f77b4",   # blue
    "cov": "#2ca02c",   # green
}

#: DANN feature-extraction variants — colour by architecture (1L vs 2L),
#: line style by training variant.
DANN_TAG_STYLE: dict[str, tuple[str, str, str]] = {
    # tag: (color, linestyle, marker)
    "baseline":        ("#7f7f7f", "-",  "o"),  # gray, solid, circle      (1L baseline)
    "dann":            ("#d62728", "-",  "s"),  # red,  solid, square      (1L DANN)
    "mtl":             ("#d62728", "--", "^"),  # red,  dashed, triangle   (1L MTL)
    "baseline_2layer": ("#2ca02c", "-",  "o"),  # green, solid, circle     (2L baseline)
    "dann_2layer":     ("#2ca02c", "-",  "s"),  # green, solid, square     (2L DANN)
    "mtl_2layer":      ("#2ca02c", "--", "^"),  # green, dashed, triangle  (2L MTL)
}

#: Project-wide per-hospital comparison palette
#: (matches the sibling repo's auc_dann_strategies plot style):
#:   gray-dotted-square (baseline), blue-dashdot-triangle (DANN raw),
#:   green-solid-circle (DANN-CF / preferred).
#: Use for any per-hospital line plot to keep visual consistency.
VARIANT_STYLE: dict[str, tuple[str, str, str]] = {
    # name: (color, linestyle, marker)
    "baseline":     ("#888888", ":",  "s"),  # gray, dotted, square
    "manual":       ("#888888", ":",  "s"),  # alias for unharmonised reference
    "raw":          ("#888888", ":",  "s"),  # alias
    "dann":         ("#1E90FF", "-.", "v"),  # blue, dash-dot, down-triangle   (1L DANN)
    "dann_raw":     ("#1E90FF", "-.", "v"),  # alias
    "mtl":          ("#8A2BE2", "--", "D"),  # purple, dashed, diamond         (1L MTL)
    "baseline_2layer": ("#2ca02c", "-", "s"),  # green, solid, square          (2L baseline)
    "dann_2layer":  ("#2ca02c", "-",  "v"),  # green, solid, down-triangle     (2L DANN)
    "mtl_2layer":   ("#2ca02c", "--", "D"),  # green, dashed, diamond          (2L MTL)
    "dann_cf":      ("#2ca02c", "-",  "o"),  # green, solid, circle
    "dann_ft":      ("#2ca02c", "-",  "o"),  # alias for fine-tuned
    "minet":        ("#888888", ":",  "s"),
    "minet_mtl":    ("#8A2BE2", "--", "D"),
    "minet_dann":   ("#1E90FF", "-.", "v"),
    "minet_dann_cf":("#2ca02c", "-",  "o"),
    # PCA conditions, when overlayed on per-hospital plots
    "pca_only":     ("#FF8C00", ":",  "s"),  # orange, dotted, square
    "pca_full":     ("#2ca02c", "-",  "o"),
}


def per_hospital_plot_kwargs(name: str, *, error_bars: bool = False) -> dict:
    """Return ready-to-splat matplotlib kwargs for a per-hospital comparison
    line/scatter using the shared VARIANT_STYLE. Falls back to default style if
    `name` is unknown."""
    color, ls, marker = VARIANT_STYLE.get(name, ("#444444", "-", "o"))
    kw = dict(color=color, linestyle=ls, marker=marker,
              markersize=8, markeredgecolor="#222222", markeredgewidth=0.7,
              linewidth=1.8, alpha=0.95, zorder=3)
    if error_bars:
        kw["capsize"] = 4
        kw["capthick"] = 1.2
        kw["elinewidth"] = 1.0
    return kw


def legend_kwargs(ncol: int = 3, **extra) -> dict:
    """Standard legend placement for result plots: upper-right, sitting just
    *above* the plotting area so it never overlaps high scores. Pair with a
    horizontal layout (ncol = number of variants) and no axes title."""
    kw = dict(loc="upper right", bbox_to_anchor=(1.0, 1.10),
              ncol=ncol, framealpha=0.95, frameon=True, borderaxespad=0.0)
    kw.update(extra)
    return kw

#: Canonical ordering for ablation conditions.
ABLATION_CONDITION_ORDER: list[str] = [
    "full_coh", "full_pow", "full_cov",
    "no_coh",   "no_pow",   "no_cov",
    "sub_coh",  "sub_pow",  "sub_cov",
]


# ---------- figsizes --------------------------------------------------------

FIGSIZE_WIDE:   tuple[float, float] = (16, 6)    # per-hospital line/bar charts
FIGSIZE_SQUARE: tuple[float, float] = (8, 7)     # scatter, single-axis plots
FIGSIZE_TALL:   tuple[float, float] = (9, 11)    # vertical multi-panel
FIGSIZE_GRID:   tuple[float, float] = (14, 10)   # multi-panel grids


# ---------- helpers ---------------------------------------------------------

def natural_sort_key(s: Any) -> list:
    """Sort H1, H2, ..., H10, H11, ... and X1..X9 naturally rather than
    lexicographically.

    >>> sorted(['H10', 'H2', 'H1'], key=natural_sort_key)
    ['H1', 'H2', 'H10']
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(s))]


def hospital_color(h_id: str, included: str = "#4C72B0",
                   excluded: str = "#9E9E9E") -> str:
    """Return colour for a hospital label — included (H_i) vs excluded (X_i)."""
    return included if str(h_id).startswith("H") else excluded


def subject_colors(subjects: Any,
                   cmaps: tuple[str, ...] = ("tab20", "tab20b")) -> dict[str, tuple]:
    """Assign each subject (e.g. hospital) a distinct, stable colour.

    Subjects are naturally sorted (H1, H2, …, H10) then mapped through the given
    qualitative colormaps in turn, so up to ~40 hospitals get unique colours and
    the mapping is reproducible across figures for the same id set. Used by the
    paired per-hospital plots to track individual sites across conditions.
    """
    subs = sorted({str(s) for s in subjects}, key=natural_sort_key)
    palette: list = []
    for name in cmaps:
        palette.extend(plt.get_cmap(name).colors)
    return {s: palette[i % len(palette)] for i, s in enumerate(subs)}
