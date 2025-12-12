"""Preprocessing module for EEG data."""

from src.data.preprocessing.pipeline import preprocess_raw_eeg, preprocess_and_segment
from src.data.preprocessing.filtering import apply_filters
from src.data.preprocessing.channel_ops import (
    standardize_electrode_names_and_order,
    apply_average_reference,
    ChannelsError
)
from src.data.preprocessing.quality_control import (
    segment_into_epochs,
    select_clean_epochs,
    compute_quality_metrics
)

__all__ = [
    'preprocess_raw_eeg',
    'preprocess_and_segment',
    'apply_filters',
    'standardize_electrode_names_and_order',
    'apply_average_reference',
    'segment_into_epochs',
    'select_clean_epochs',
    'compute_quality_metrics',
    'ChannelsError',
]
