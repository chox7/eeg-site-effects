"""
Connectivity feature extraction from EEG data.

This module computes connectivity features (coherence, phase locking) between channels.
"""

import mne
import numpy as np
import mne_connectivity

from src.utils.utils import FREQ_BANDS_PH
from src.data.config.preprocessing_configs import FeatureExtractionConfig


def compute_connectivity_features(
    segments: np.ndarray,
    sampling_freq: float,
    config: FeatureExtractionConfig
) -> np.ndarray:
    """
    Compute connectivity features (coherence) between EEG channels.

    Args:
        segments: EEG segments (n_epochs, n_channels, n_samples)
        sampling_freq: Sampling frequency in Hz
        config: FeatureExtractionConfig specifying extraction parameters

    Returns:
        connectivity_features: Array of shape (n_epochs, n_bands * n_connections)
                              where n_connections = n_channels * (n_channels - 1) / 2
    """
    mne.set_log_level("CRITICAL")

    n_channels = segments.shape[1]
    n_samples = segments.shape[2]
    n_bands = len(FREQ_BANDS_PH)
    n_connections = int(n_channels * (n_channels - 1) / 2)

    connectivity_features = np.zeros((len(segments), n_bands, n_connections))

    for i, segment in enumerate(segments):
        # Split segment into fragments for better spectral estimation
        n_fragments = n_samples // int(sampling_freq)
        fragment_length = n_samples // n_fragments

        fragments = np.zeros((n_fragments, n_channels, fragment_length))
        start = 0
        for frag_idx in range(n_fragments):
            fragments[frag_idx, :, :] = segment[:, start:start + fragment_length]
            start += fragment_length

        # Compute spectral connectivity
        connectivity = mne_connectivity.spectral_connectivity_epochs(
            fragments,
            method=config.connectivity_method,
            sfreq=sampling_freq,
            mode=config.connectivity_mode,
            faverage=True,
            fmin=FREQ_BANDS_PH[:, 0],
            fmax=FREQ_BANDS_PH[:, 1]
        )

        # Extract connectivity matrix
        conn_matrix = connectivity.get_data('dense')

        # Extract lower triangular part for each frequency band
        for band_idx in range(n_bands):
            band_connectivity = conn_matrix[:, :, band_idx]
            connectivity_features[i, band_idx, :] = band_connectivity[
                np.tril_indices(n_channels, k=-1)
            ]

    # Reshape to (n_epochs, n_bands * n_connections)
    connectivity_features = connectivity_features.reshape(
        connectivity_features.shape[0],
        connectivity_features.shape[1] * connectivity_features.shape[2]
    )

    return connectivity_features


def aggregate_connectivity_features(
    connectivity_features: np.ndarray,
    method: str = 'median'
) -> np.ndarray:
    """
    Aggregate connectivity features across segments.

    Args:
        connectivity_features: Array of shape (n_epochs, n_features)
        method: Aggregation method ('median' or 'mean')

    Returns:
        Aggregated connectivity features of shape (n_features,)
    """
    if method == 'median':
        return np.median(connectivity_features, axis=0)
    elif method == 'mean':
        return np.mean(connectivity_features, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
