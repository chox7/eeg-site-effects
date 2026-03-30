"""
Utility modules for EEG site effect analysis.

This package provides:
- cv_metrics: Cross-validation metrics and stratified fold generation
- statistics: Statistical functions (Cohen's d, effect sizes)
- eeg_constants: EEG channel and frequency band constants
- utils: Legacy utilities (channel mappings, feature naming)
"""

# Import constants (no external dependencies)
from src.utils.eeg_constants import (
    CHANNELS_19,
    CHANNEL_POSITIONS_19,
    FREQ_BANDS_14,
    FREQ_BAND_NAMES_14,
    FREQ_BANDS_NAMED,
)

# Import statistics (numpy only)
from src.utils.statistics import (
    cohens_d,
    compute_effect_sizes_per_site,
)

# Import cv_metrics (requires sklearn)
try:
    from src.utils.cv_metrics import (
        get_scores_multiclass,
        get_scores_binary,
        compute_confusion_matrix,
        stratified_site_folds,
    )
except ImportError:
    # sklearn not installed - cv_metrics functions unavailable
    pass

# Import data_prep utilities
from src.utils.data_prep import (
    load_experiment_data,
    prepare_pathology_labels,
    append_results_csv,
    PATHOLOGY_LABEL_MAP,
    COLUMN_RENAME_MAP,
)

__all__ = [
    # eeg_constants
    "CHANNELS_19",
    "CHANNEL_POSITIONS_19",
    "FREQ_BANDS_14",
    "FREQ_BAND_NAMES_14",
    "FREQ_BANDS_NAMED",
    # statistics
    "cohens_d",
    "compute_effect_sizes_per_site",
    # cv_metrics (if sklearn available)
    "get_scores_multiclass",
    "get_scores_binary",
    "compute_confusion_matrix",
    "stratified_site_folds",
    # data_prep
    "load_experiment_data",
    "prepare_pathology_labels",
    "append_results_csv",
    "PATHOLOGY_LABEL_MAP",
    "COLUMN_RENAME_MAP",
]
