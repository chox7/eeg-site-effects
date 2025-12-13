"""
Covariance feature extraction from EEG data.

This module computes covariance matrices from EEG segments.
"""

import numpy as np
from pyriemann.estimation import Covariances

from src.utils.utils import CH_NAMES


def compute_covariance_features(segments: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrices for EEG segments.

    Args:
        segments: EEG segments (n_epochs, n_channels, n_samples)

    Returns:
        covariance_features: Array of shape (n_epochs, n_channels, n_channels)
                            containing covariance matrices
    """
    cov_estimator = Covariances()
    covariance_matrices = cov_estimator.fit_transform(segments)
    return covariance_matrices


def aggregate_covariance_features(
    covariance_matrices: np.ndarray,
    method: str = 'median'
) -> np.ndarray:
    """
    Aggregate covariance matrices across segments and extract lower triangular part.

    Args:
        covariance_matrices: Array of shape (n_epochs, n_channels, n_channels)
        method: Aggregation method ('median' or 'mean')

    Returns:
        Aggregated covariance features as 1D array (lower triangular elements including diagonal)
    """
    # Aggregate across epochs
    if method == 'median':
        aggregated = np.median(covariance_matrices, axis=0)
    elif method == 'mean':
        aggregated = np.mean(covariance_matrices, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    # Extract lower triangular part (including diagonal)
    n_channels = len(CH_NAMES)
    covariance_features = aggregated[np.tril_indices(n_channels, k=0)]

    return covariance_features
