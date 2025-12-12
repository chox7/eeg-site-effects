"""
Main preprocessing pipeline orchestrator.

This module provides high-level functions to run the complete EEG preprocessing pipeline.
"""

import mne
import numpy as np
from typing import Tuple, Optional

from src.data.config.preprocessing_configs import PreprocessingConfig
from src.data.preprocessing.filtering import apply_filters
from src.data.preprocessing.channel_ops import (
    standardize_electrode_names_and_order,
    apply_average_reference,
    set_standard_montage
)
from src.data.preprocessing.quality_control import (
    segment_into_epochs,
    select_clean_epochs
)


def preprocess_raw_eeg(
    edf: mne.io.Raw,
    config: PreprocessingConfig
) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """
    Apply complete preprocessing pipeline to raw EEG data.

    Pipeline steps:
    1. Standardize channel names and order
    2. Set standard 10-20 montage
    3. Apply filters (highpass, lowpass, notch)
    4. Apply average reference (if configured)

    Args:
        edf: MNE Raw object with raw EEG data
        config: PreprocessingConfig specifying preprocessing parameters

    Returns:
        edf_without_ref: Preprocessed data without average reference
        edf_with_ref: Preprocessed data with average reference applied

    Raises:
        ChannelsError: If channel standardization fails
        BadCoefficients: If filter design fails
    """
    with mne.utils.use_log_level("error"):
        # Standardize channels
        edf = standardize_electrode_names_and_order(edf)
        edf = set_standard_montage(edf)

        # Apply filters
        edf = apply_filters(edf, config.filters)

        # Create version with and without average reference
        edf_without_ref = edf.copy()

        if config.apply_average_reference:
            edf_with_ref = apply_average_reference(edf.copy())
        else:
            edf_with_ref = edf.copy()

    return edf_without_ref, edf_with_ref


def preprocess_and_segment(
    edf: mne.io.Raw,
    config: PreprocessingConfig
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[mne.io.Raw]]:
    """
    Complete preprocessing pipeline: preprocess, segment, and quality control.

    Args:
        edf: MNE Raw object with raw EEG data
        config: PreprocessingConfig specifying all parameters

    Returns:
        clean_segments: Array of clean EEG segments (n_epochs, n_channels, n_samples)
        time_indices: Time indices for each clean segment
        preprocessed_edf: Preprocessed MNE Raw object (with reference)

    Returns (None, None, None) if preprocessing fails.
    """
    try:
        # Preprocess
        edf_without_ref, edf_with_ref = preprocess_raw_eeg(edf, config)

        # Segment into epochs
        crops_with_ref, time_indices = segment_into_epochs(
            edf_with_ref,
            config.quality_control.crop_length_samples,
            return_indices=True
        )

        crops_without_ref = segment_into_epochs(
            edf_without_ref,
            config.quality_control.crop_length_samples,
            return_indices=False
        )

        # Quality control - select clean epochs
        clean_segments, clean_time_indices = select_clean_epochs(
            crops_with_ref,
            crops_without_ref,
            time_indices,
            config.quality_control
        )

        return clean_segments, clean_time_indices, edf_with_ref

    except Exception as e:
        print(f"[WARNING] Preprocessing failed: {e}")
        return None, None, None
