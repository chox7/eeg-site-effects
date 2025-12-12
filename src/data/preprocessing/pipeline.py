"""
Main preprocessing pipeline orchestrator.

This module provides high-level functions to run the complete EEG preprocessing pipeline.
"""

import mne
import numpy as np
import time
from typing import Tuple, Optional
from datetime import datetime

from src.data.config.preprocessing_configs import PreprocessingConfig
from src.data.preprocessing.filtering import apply_filters
from src.data.preprocessing.channel_ops import (
    standardize_electrode_names_and_order,
    apply_average_reference,
    set_standard_montage
)
from src.data.preprocessing.quality_control import (
    segment_into_epochs,
    select_clean_epochs,
    detect_artifacts
)
from src.data.preprocessing.quality_metrics import (
    PreprocessingMetrics,
    compute_signal_quality_metrics,
    create_segmentation_metrics
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


def preprocess_and_segment_with_metrics(
    edf: mne.io.Raw,
    config: PreprocessingConfig,
    examination_id: str,
    institution_id: str,
    data_group: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[mne.io.Raw], PreprocessingMetrics]:
    """
    Complete preprocessing pipeline with comprehensive quality metrics.

    Args:
        edf: MNE Raw object with raw EEG data
        config: PreprocessingConfig specifying all parameters
        examination_id: Examination identifier
        institution_id: Institution/hospital identifier
        data_group: Dataset name

    Returns:
        clean_segments: Array of clean EEG segments (n_epochs, n_channels, n_samples)
        time_indices: Time indices for each clean segment
        preprocessed_edf: Preprocessed MNE Raw object (with reference)
        metrics: PreprocessingMetrics object with comprehensive quality metrics
    """
    start_time = time.time()

    # Initialize metrics
    metrics = PreprocessingMetrics(
        examination_id=examination_id,
        institution_id=institution_id,
        data_group=data_group,
        processing_timestamp=datetime.now().isoformat(),
        sampling_frequency_hz=edf.info['sfreq'],
        recording_duration_seconds=edf.times[-1],
        n_channels=len(edf.ch_names)
    )

    # Track filters applied
    metrics.filters_applied = [f"{f.type}" for f in config.filters]

    try:
        # Compute raw signal quality
        raw_data = edf.get_data(units="uV")
        metrics.raw_signal_quality = compute_signal_quality_metrics(raw_data)

        # Preprocess
        edf_without_ref, edf_with_ref = preprocess_raw_eeg(edf, config)

        # Compute preprocessed signal quality
        preprocessed_data = edf_with_ref.get_data(units="uV")
        metrics.preprocessed_signal_quality = compute_signal_quality_metrics(preprocessed_data)

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

        # Quality control - detect artifacts and get statistics
        reject_max, reject_min, reject = detect_artifacts(
            crops_with_ref,
            crops_without_ref,
            config.quality_control
        )

        # Create segmentation metrics
        metrics.segmentation = create_segmentation_metrics(
            total_segments=len(crops_with_ref),
            rejected_amplitude=int(np.sum(reject_max)),
            rejected_flat=int(np.sum(reject_min)),
            rejected_total=int(np.sum(reject)),
            kept=int(np.sum(~reject))
        )

        # Add warnings
        if metrics.segmentation.rejection_rate > 0.5:
            metrics.add_warning(f"High rejection rate: {metrics.segmentation.rejection_rate:.1%}")

        if metrics.segmentation.segments_kept == 0:
            metrics.add_warning("No clean segments remaining after quality control")

        # Select clean epochs
        clean_segments = crops_with_ref[~reject]
        clean_time_indices = list(time_indices[~reject])

        # Record processing time
        metrics.processing_duration_seconds = time.time() - start_time
        metrics.preprocessing_successful = True

        return clean_segments, clean_time_indices, edf_with_ref, metrics

    except Exception as e:
        metrics.preprocessing_successful = False
        metrics.failure_reason = str(e)
        metrics.processing_duration_seconds = time.time() - start_time
        print(f"[WARNING] Preprocessing failed: {e}")
        return None, None, None, metrics
