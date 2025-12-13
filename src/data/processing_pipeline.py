"""
Main feature extraction interface.

This module provides high-level functions for extracting features from EEG files.
"""

import mne
import numpy as np
import glob
from typing import Optional, Tuple

from src.utils.utils import apply_mor_data_hack_fix
from src.data.config.preprocessing_configs import (
    PreprocessingConfig,
    FeatureExtractionConfig,
    DEFAULT_PREPROCESSING_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    get_site_specific_config
)
from src.data.preprocessing import (
    preprocess_and_segment,
    preprocess_and_segment_with_metrics,
    ChannelsError,
    PreprocessingMetrics
)
from src.data.feature_extraction import extract_all_features


def get_edf_path(examination_id: str) -> str:
    """
    Get EDF file path for TUH dataset.

    Args:
        examination_id: Examination identifier

    Returns:
        Path to EDF file
    """
    base_path = "datasets/gemein/raw/gemein/raw/*/*/01_tcp_ar/"
    patient_id, session_id, t_id = examination_id.split("_")
    patient_group = patient_id[3:6]
    path_pattern = base_path + f"{patient_group}/{patient_id}/{session_id}_*/{examination_id}.edf"
    files = glob.glob(path_pattern)
    return files[0]


def map_edf_to_samples(
    examination_id: str,
    institution_id: str,
    edf_dir_path: str,
    data_group: str,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    feature_config: Optional[FeatureExtractionConfig] = None
) -> Optional[np.ndarray]:
    """
    Process an EEG file and extract features.

    Args:
        examination_id: Examination identifier
        institution_id: Institution/hospital identifier
        edf_dir_path: Directory containing EDF files
        data_group: Dataset name ('ELM19' or 'TUH')
        preprocessing_config: Optional preprocessing configuration (if None, uses site-specific config)
        feature_config: Optional feature extraction configuration (if None, uses defaults)

    Returns:
        Feature vector as 1D numpy array, or None if processing fails
    """
    mne.set_log_level("CRITICAL")

    # Use site-specific config if not provided
    if preprocessing_config is None:
        preprocessing_config = get_site_specific_config(institution_id)

    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG

    # Get EDF file path
    if data_group == "ELM19":
        edf_path = f"{edf_dir_path}/{examination_id}.edf"
    elif data_group == "TUH":
        edf_path = get_edf_path(examination_id)
    else:
        raise ValueError(f"Unsupported data_group: {data_group}")

    # Load EDF file
    raw_edf = mne.io.read_raw_edf(edf_path, preload=True)

    # Apply MOR hospital data fix if needed
    if institution_id == 'MOR':
        raw_edf = apply_mor_data_hack_fix(raw_edf, edf_path, institution_id)

    # Preprocess and segment
    try:
        clean_segments, time_indices, preprocessed_edf = preprocess_and_segment(
            raw_edf,
            preprocessing_config
        )
    except (ValueError, ChannelsError) as e:
        print(f"[WARNING] Skipping due to error: {e}")
        return None

    if clean_segments is None or len(clean_segments) == 0:
        print(f"[WARNING] No clean segments after quality control")
        return None

    # Extract features
    features = extract_all_features(
        clean_segments,
        preprocessing_config.desired_sampling_freq,
        feature_config
    )

    return features


def map_edf_to_samples_with_idx(
    examination_id: str,
    institution_id: str,
    idx: int,
    edf_dir_path: str,
    data_group: str = "ELM19",
    preprocessing_config: Optional[PreprocessingConfig] = None,
    feature_config: Optional[FeatureExtractionConfig] = None
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Process an EEG file and extract features, returning index along with result.

    Useful for parallel processing to maintain order of results.

    Args:
        examination_id: Examination identifier
        institution_id: Institution/hospital identifier
        idx: Index for tracking (useful in parallel processing)
        edf_dir_path: Directory containing EDF files
        data_group: Dataset name ('ELM19' or 'TUH')
        preprocessing_config: Optional preprocessing configuration
        feature_config: Optional feature extraction configuration

    Returns:
        Tuple of (idx, features)
    """
    features = map_edf_to_samples(
        examination_id,
        institution_id,
        edf_dir_path,
        data_group,
        preprocessing_config,
        feature_config
    )
    return idx, features


def map_edf_to_samples_with_metrics(
    examination_id: str,
    institution_id: str,
    edf_dir_path: str,
    data_group: str,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    feature_config: Optional[FeatureExtractionConfig] = None
) -> Tuple[Optional[np.ndarray], PreprocessingMetrics]:
    """
    Process an EEG file and extract features with comprehensive quality metrics.

    This function is crucial for understanding site-specific preprocessing differences.

    Args:
        examination_id: Examination identifier
        institution_id: Institution/hospital identifier
        edf_dir_path: Directory containing EDF files
        data_group: Dataset name ('ELM19' or 'TUH')
        preprocessing_config: Optional preprocessing configuration
        feature_config: Optional feature extraction configuration

    Returns:
        Tuple of (features, metrics)
        - features: 1D numpy array or None if processing fails
        - metrics: PreprocessingMetrics object with comprehensive quality information
    """
    mne.set_log_level("CRITICAL")

    # Use site-specific config if not provided
    if preprocessing_config is None:
        preprocessing_config = get_site_specific_config(institution_id)

    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG

    # Get EDF file path
    if data_group == "ELM19":
        edf_path = f"{edf_dir_path}/{examination_id}.edf"
    elif data_group == "TUH":
        edf_path = get_edf_path(examination_id)
    else:
        raise ValueError(f"Unsupported data_group: {data_group}")

    # Load EDF file
    raw_edf = mne.io.read_raw_edf(edf_path, preload=True)

    # Apply MOR hospital data fix if needed
    if institution_id == 'MOR':
        raw_edf = apply_mor_data_hack_fix(raw_edf, edf_path, institution_id)

    # Preprocess and segment with metrics
    clean_segments, time_indices, preprocessed_edf, metrics = preprocess_and_segment_with_metrics(
        raw_edf,
        preprocessing_config,
        examination_id,
        institution_id,
        data_group
    )

    # Extract features if preprocessing was successful
    if clean_segments is not None and len(clean_segments) > 0:
        features = extract_all_features(
            clean_segments,
            preprocessing_config.desired_sampling_freq,
            feature_config
        )
    else:
        features = None
        if metrics.preprocessing_successful:
            metrics.add_warning("No features extracted - no clean segments available")

    return features, metrics
