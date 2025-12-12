"""Feature extraction module for EEG data."""

from src.data.feature_extraction.pipeline import extract_all_features
from src.data.feature_extraction.spectral import (
    compute_power_features,
    aggregate_power_features
)
from src.data.feature_extraction.connectivity import (
    compute_connectivity_features,
    aggregate_connectivity_features
)
from src.data.feature_extraction.covariance import (
    compute_covariance_features,
    aggregate_covariance_features
)

__all__ = [
    'extract_all_features',
    'compute_power_features',
    'aggregate_power_features',
    'compute_connectivity_features',
    'aggregate_connectivity_features',
    'compute_covariance_features',
    'aggregate_covariance_features',
]
