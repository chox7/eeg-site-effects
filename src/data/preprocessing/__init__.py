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
from src.data.preprocessing.visualization import (
    plot_signal_comparison,
    plot_power_spectrum_comparison,
    plot_preprocessing_summary,
    save_preprocessing_report
)
from src.data.preprocessing.site_comparison import (
    plot_rejection_rates_by_site,
    plot_signal_quality_by_site,
    plot_site_comparison_dashboard,
    plot_metric_distribution_by_site
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
    'plot_signal_comparison',
    'plot_power_spectrum_comparison',
    'plot_preprocessing_summary',
    'save_preprocessing_report',
    'plot_rejection_rates_by_site',
    'plot_signal_quality_by_site',
    'plot_site_comparison_dashboard',
    'plot_metric_distribution_by_site',
    'ChannelsError',
]
