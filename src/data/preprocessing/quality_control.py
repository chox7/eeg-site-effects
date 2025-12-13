"""
Quality control and artifact rejection for EEG preprocessing.

This module handles segmentation, artifact detection, and quality assessment.
"""

import mne
import numpy as np
import logging
from typing import Tuple, List

from src.data.config.preprocessing_configs import QualityControlConfig


logger = logging.getLogger(__name__)


def segment_into_epochs(
    edf: mne.io.Raw,
    crop_length_samples: int,
    return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Segment continuous EEG data into fixed-length epochs.

    Args:
        edf: MNE Raw object
        crop_length_samples: Length of each segment in samples
        return_indices: If True, also return starting indices of each segment

    Returns:
        crops: Array of shape (n_epochs, n_channels, crop_length_samples)
        indices: (optional) Starting indices of each segment
    """
    data = edf.get_data(units="uV")

    tmax = data.shape[-1]
    num_crops = tmax // crop_length_samples

    crops = np.zeros((num_crops, data.shape[0], crop_length_samples))
    indices = np.arange(0, num_crops * crop_length_samples, crop_length_samples)

    for i, idx in enumerate(indices):
        crops[i, :, :] = data[:, idx:idx + crop_length_samples]

    if return_indices:
        return crops, indices
    else:
        return crops


def detect_artifacts(
    crops_with_ref: np.ndarray,
    crops_without_ref: np.ndarray,
    config: QualityControlConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect artifacts in EEG segments based on amplitude and flatness criteria.

    Args:
        crops_with_ref: Segments with average reference applied (n_epochs, n_channels, n_samples)
        crops_without_ref: Segments without reference (for flatness detection)
        config: QualityControlConfig with rejection thresholds

    Returns:
        reject_max: Boolean array indicating segments exceeding max amplitude
        reject_min: Boolean array indicating flat segments
        reject: Boolean array indicating all rejected segments
    """
    # Detect segments with excessive amplitude
    reject_max = np.sum(
        np.abs(crops_with_ref) > config.max_amplitude_uv,
        axis=(-1, -2)
    )
    reject_max = np.where(reject_max != 0, 1, 0).astype(bool)

    # Detect flat channels using MAD (Median Absolute Deviation)
    mad = np.median(
        np.abs(crops_without_ref - np.median(crops_without_ref, axis=-1, keepdims=True)),
        axis=-1
    )
    min_mad = np.min(mad, axis=-1)
    reject_min = min_mad < config.min_mad_uv

    # Combine rejection criteria
    reject = np.logical_or(reject_min, reject_max)

    return reject_max, reject_min, reject


def select_clean_epochs(
    crops_with_ref: np.ndarray,
    crops_without_ref: np.ndarray,
    time_indices: np.ndarray,
    config: QualityControlConfig
) -> Tuple[np.ndarray, List[int]]:
    """
    Select clean EEG epochs by rejecting artifacts.

    Args:
        crops_with_ref: Segments with average reference applied
        crops_without_ref: Segments without reference
        time_indices: Starting time indices for each segment
        config: QualityControlConfig with rejection thresholds

    Returns:
        clean_crops: Array of clean segments
        clean_indices: List of time indices for clean segments
    """
    reject_max, reject_min, reject = detect_artifacts(
        crops_with_ref,
        crops_without_ref,
        config
    )

    not_reject = np.logical_not(reject)

    logger.info(
        f"Total: {len(reject):3d} | "
        f"Rejected - max: {np.sum(reject_max):3d}, "
        f"min: {np.sum(reject_min):3d}, "
        f"total: {np.sum(reject):3d} | "
        f"Kept: {np.sum(not_reject):3d}"
    )

    clean_crops = crops_with_ref[not_reject]
    clean_indices = list(time_indices[not_reject])

    return clean_crops, clean_indices


def compute_quality_metrics(
    crops: np.ndarray,
    config: QualityControlConfig
) -> dict:
    """
    Compute quality metrics for EEG segments.

    Args:
        crops: EEG segments (n_epochs, n_channels, n_samples)
        config: QualityControlConfig

    Returns:
        Dictionary containing quality metrics
    """
    reject_max, reject_min, reject = detect_artifacts(crops, crops, config)

    metrics = {
        'total_segments': len(crops),
        'rejected_amplitude': int(np.sum(reject_max)),
        'rejected_flat': int(np.sum(reject_min)),
        'total_rejected': int(np.sum(reject)),
        'total_clean': int(np.sum(~reject)),
        'rejection_rate': float(np.sum(reject) / len(crops))
    }

    return metrics
