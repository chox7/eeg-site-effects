"""
Spectral feature extraction from EEG data.

This module computes power spectral density features across frequency bands.
"""

import mne
import numpy as np
import mne_features

from src.utils.utils import FREQ_BANDS
from src.data.config.preprocessing_configs import FeatureExtractionConfig


def compute_power_features(
    segments: np.ndarray,
    sampling_freq: float,
    config: FeatureExtractionConfig
) -> np.ndarray:
    """
    Compute power spectral density features for EEG segments.

    Args:
        segments: EEG segments (n_epochs, n_channels, n_samples)
        sampling_freq: Sampling frequency in Hz
        config: FeatureExtractionConfig specifying extraction parameters

    Returns:
        power_features: Array of shape (n_epochs, n_channels * n_bands)
                       containing normalized power features
    """
    mne.set_log_level("CRITICAL")

    n_channels = segments.shape[1]
    n_bands = len(FREQ_BANDS)
    power_features = np.zeros((len(segments), n_channels * n_bands))

    for i, segment in enumerate(segments):
        # Compute power in each frequency band
        powers = mne_features.univariate.compute_pow_freq_bands(
            sampling_freq,
            segment,
            FREQ_BANDS,
            normalize=False,
            psd_method=config.psd_method
        )

        powers = powers.reshape((n_channels, n_bands))

        # Normalize power across all bands and channels
        if config.normalize_power:
            norm = np.sum(powers)
            powers = powers / norm

        power_features[i] = powers.flatten()

    return power_features


def aggregate_power_features(
    power_features: np.ndarray,
    method: str = 'median'
) -> np.ndarray:
    """
    Aggregate power features across segments.

    Args:
        power_features: Array of shape (n_epochs, n_features)
        method: Aggregation method ('median' or 'mean')

    Returns:
        Aggregated power features of shape (n_features,)
    """
    if method == 'median':
        return np.median(power_features, axis=0)
    elif method == 'mean':
        return np.mean(power_features, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
