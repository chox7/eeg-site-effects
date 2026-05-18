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
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.titlesize": 14,
    # Axes / grid
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "grid.linewidth":    0.6,
    # Figure / save
    "figure.dpi":     100,
    "savefig.dpi":    150,
    "savefig.bbox":   "tight",
    "savefig.facecolor": "white",
    # Lines
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
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
    "baseline":    ("#7f7f7f", "-",  "o"),  # gray, solid, circle
    "dann":        ("#d62728", "-",  "s"),  # red,  solid, square      (1L DANN)
    "mtl":         ("#d62728", "--", "^"),  # red,  dashed, triangle   (1L MTL)
    "dann_2layer": ("#2ca02c", "-",  "s"),  # green, solid, square     (2L DANN)
    "mtl_2layer":  ("#2ca02c", "--", "^"),  # green, dashed, triangle  (2L MTL)
}

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
