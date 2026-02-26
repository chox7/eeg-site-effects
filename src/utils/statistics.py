"""
Statistical utility functions for EEG site effect analysis.
"""

import numpy as np


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two means.
    Uses pooled standard deviation for the denominator.

    Parameters
    ----------
    x1 : array-like
        First group of observations.
    x2 : array-like
        Second group of observations.

    Returns
    -------
    float
        Cohen's d effect size. Returns 0 if pooled std is 0.

    Notes
    -----
    Effect size interpretation (Cohen, 1988):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    denom = n1 + n2 - 2
    if denom == 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / denom)

    if pooled_std == 0:
        return 0.0

    return (m1 - m2) / pooled_std


def compute_effect_sizes_per_site(
    df_features: "pd.DataFrame",
    df_info: "pd.DataFrame",
    site_column: str = "institution_id",
) -> "pd.DataFrame":
    """
    Compute Cohen's d effect sizes comparing each site to all others.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix with samples as rows.
    df_info : pd.DataFrame
        Metadata with site information.
    site_column : str, default="institution_id"
        Column containing site identifiers.

    Returns
    -------
    pd.DataFrame
        Effect sizes with sites as rows and features as columns.
    """
    import pandas as pd

    df_all = pd.concat(
        [df_info.reset_index(drop=True), df_features.reset_index(drop=True)],
        axis=1,
    )

    results = {}
    for site_id, group in df_all.groupby(site_column):
        rest = df_all.loc[df_all[site_column] != site_id, df_features.columns]
        d_scores = {
            feat: cohens_d(group[feat].values, rest[feat].values)
            for feat in df_features.columns
        }
        results[site_id] = d_scores

    return pd.DataFrame(results).T
