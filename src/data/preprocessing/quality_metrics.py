"""
Quality metrics collection for EEG preprocessing.

This module provides comprehensive quality metrics to track preprocessing
performance and identify site-specific differences.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class SignalQualityMetrics:
    """Metrics describing signal quality."""
    mean_amplitude_uv: float
    std_amplitude_uv: float
    max_amplitude_uv: float
    min_amplitude_uv: float
    median_mad_uv: float  # Median of MAD across channels


@dataclass
class SegmentationMetrics:
    """Metrics from segmentation and quality control."""
    total_segments: int
    segments_rejected_amplitude: int
    segments_rejected_flat: int
    segments_rejected_total: int
    segments_kept: int
    rejection_rate: float


@dataclass
class PreprocessingMetrics:
    """Comprehensive preprocessing quality metrics.

    These metrics help identify site-specific preprocessing issues
    and track data quality across different hospitals.
    """
    # File identifiers
    examination_id: str
    institution_id: str
    data_group: str

    # Timing
    processing_timestamp: str

    # Signal properties
    sampling_frequency_hz: float
    recording_duration_seconds: float
    n_channels: int

    # Signal quality (raw)
    raw_signal_quality: Optional[SignalQualityMetrics] = None

    # Signal quality (after preprocessing)
    preprocessed_signal_quality: Optional[SignalQualityMetrics] = None

    # Segmentation and QC
    segmentation: Optional[SegmentationMetrics] = None

    # Filter settings applied
    filters_applied: List[str] = field(default_factory=list)

    # Warnings/issues
    warnings: List[str] = field(default_factory=list)

    # Success flag
    preprocessing_successful: bool = True
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    @property
    def quality_summary(self) -> str:
        """Get a human-readable quality summary."""
        if not self.preprocessing_successful:
            return f"FAILED: {self.failure_reason}"

        seg = self.segmentation
        if seg:
            return (
                f"Site: {self.institution_id} | "
                f"Segments: {seg.segments_kept}/{seg.total_segments} "
                f"({seg.rejection_rate:.1%} rejected) | "
                f"Warnings: {len(self.warnings)}"
            )
        return "No segmentation performed"


def compute_signal_quality_metrics(data: np.ndarray) -> SignalQualityMetrics:
    """
    Compute signal quality metrics from EEG data.

    Args:
        data: EEG data (n_channels, n_samples) in µV

    Returns:
        SignalQualityMetrics object
    """
    # Compute MAD per channel
    mad_per_channel = np.median(
        np.abs(data - np.median(data, axis=-1, keepdims=True)),
        axis=-1
    )

    return SignalQualityMetrics(
        mean_amplitude_uv=float(np.mean(np.abs(data))),
        std_amplitude_uv=float(np.std(data)),
        max_amplitude_uv=float(np.max(np.abs(data))),
        min_amplitude_uv=float(np.min(np.abs(data))),
        median_mad_uv=float(np.median(mad_per_channel))
    )


def create_segmentation_metrics(
    total_segments: int,
    rejected_amplitude: int,
    rejected_flat: int,
    rejected_total: int,
    kept: int
) -> SegmentationMetrics:
    """Create segmentation metrics object."""
    rejection_rate = rejected_total / total_segments if total_segments > 0 else 0.0

    return SegmentationMetrics(
        total_segments=total_segments,
        segments_rejected_amplitude=rejected_amplitude,
        segments_rejected_flat=rejected_flat,
        segments_rejected_total=rejected_total,
        segments_kept=kept,
        rejection_rate=rejection_rate
    )


class MetricsAggregator:
    """Aggregates metrics across multiple files for site-level analysis."""

    def __init__(self):
        self.metrics_list: List[PreprocessingMetrics] = []

    def add_metrics(self, metrics: PreprocessingMetrics):
        """Add metrics for a single file."""
        self.metrics_list.append(metrics)

    def get_site_summary(self, site_id: str) -> Dict:
        """
        Get aggregated summary statistics for a specific site.

        Args:
            site_id: Institution identifier

        Returns:
            Dictionary with aggregated statistics
        """
        site_metrics = [m for m in self.metrics_list if m.institution_id == site_id]

        if not site_metrics:
            return {"error": f"No metrics found for site {site_id}"}

        successful = [m for m in site_metrics if m.preprocessing_successful]
        failed = [m for m in site_metrics if not m.preprocessing_successful]

        # Aggregate rejection rates
        rejection_rates = [
            m.segmentation.rejection_rate
            for m in successful
            if m.segmentation is not None
        ]

        # Aggregate signal quality
        mean_amplitudes = [
            m.preprocessed_signal_quality.mean_amplitude_uv
            for m in successful
            if m.preprocessed_signal_quality is not None
        ]

        return {
            "site_id": site_id,
            "total_files": len(site_metrics),
            "successful": len(successful),
            "failed": len(failed),
            "failure_rate": len(failed) / len(site_metrics) if site_metrics else 0,
            "rejection_rate_mean": float(np.mean(rejection_rates)) if rejection_rates else None,
            "rejection_rate_std": float(np.std(rejection_rates)) if rejection_rates else None,
            "mean_amplitude_mean": float(np.mean(mean_amplitudes)) if mean_amplitudes else None,
            "mean_amplitude_std": float(np.std(mean_amplitudes)) if mean_amplitudes else None,
            "total_warnings": sum(len(m.warnings) for m in site_metrics),
        }

    def get_all_sites_summary(self) -> Dict[str, Dict]:
        """Get summary for all sites."""
        sites = set(m.institution_id for m in self.metrics_list)
        return {site: self.get_site_summary(site) for site in sites}

    def save_all_metrics(self, filepath: str):
        """Save all metrics to JSON file."""
        metrics_dicts = [m.to_dict() for m in self.metrics_list]
        with open(filepath, 'w') as f:
            json.dump(metrics_dicts, f, indent=2)

    def load_metrics(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            metrics_dicts = json.load(f)

        # Note: This creates dicts, not full PreprocessingMetrics objects
        # You'd need to add proper deserialization if needed
        return metrics_dicts
