"""Preprocessing module for EEG data."""

from src.data.preprocessing.pipeline import (
    preprocess_raw_eeg,
    preprocess_and_segment,
    preprocess_and_segment_with_metrics
)
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
from src.data.preprocessing.quality_metrics import (
    PreprocessingMetrics,
    MetricsAggregator
)
from src.data.preprocessing.quality_report import (
    generate_text_summary,
    metrics_to_dataframe,
    save_site_comparison_report,
    print_site_comparison
)

__all__ = [
    'preprocess_raw_eeg',
    'preprocess_and_segment',
    'preprocess_and_segment_with_metrics',
    'apply_filters',
    'standardize_electrode_names_and_order',
    'apply_average_reference',
    'segment_into_epochs',
    'select_clean_epochs',
    'compute_quality_metrics',
    'PreprocessingMetrics',
    'MetricsAggregator',
    'generate_text_summary',
    'metrics_to_dataframe',
    'save_site_comparison_report',
    'print_site_comparison',
    'ChannelsError',
]
