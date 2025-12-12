"""
Feature extraction pipeline orchestrator.

This module provides high-level functions to extract all feature types from preprocessed EEG segments.
"""

import numpy as np
from typing import Optional

from src.data.config.preprocessing_configs import FeatureExtractionConfig
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


def extract_all_features(
    segments: np.ndarray,
    sampling_freq: float,
    config: FeatureExtractionConfig
) -> Optional[np.ndarray]:
    """
    Extract all configured features from EEG segments and aggregate.

    Args:
        segments: Clean EEG segments (n_epochs, n_channels, n_samples)
        sampling_freq: Sampling frequency in Hz
        config: FeatureExtractionConfig specifying which features to extract

    Returns:
        feature_vector: 1D array containing all extracted and aggregated features
                       Features are concatenated in order: [connectivity, power, covariance]
                       Returns None if no segments are provided
    """
    if len(segments) == 0:
        return None

    feature_parts = []

    # Extract connectivity features
    if config.extract_connectivity:
        connectivity = compute_connectivity_features(segments, sampling_freq, config)
        connectivity_agg = aggregate_connectivity_features(
            connectivity,
            config.aggregation_method
        )
        feature_parts.append(connectivity_agg)

    # Extract power features
    if config.extract_power:
        power = compute_power_features(segments, sampling_freq, config)
        power_agg = aggregate_power_features(power, config.aggregation_method)
        feature_parts.append(power_agg)

    # Extract covariance features
    if config.extract_covariance:
        covariance = compute_covariance_features(segments)
        covariance_agg = aggregate_covariance_features(
            covariance,
            config.aggregation_method
        )
        feature_parts.append(covariance_agg)

    # Concatenate all features
    if feature_parts:
        return np.hstack(feature_parts)
    else:
        return None
