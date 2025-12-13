"""
Quality report generation for preprocessing metrics.

This module provides utilities to generate human-readable reports from preprocessing metrics.
"""

import pandas as pd
from typing import List
from pathlib import Path

from src.data.preprocessing.quality_metrics import PreprocessingMetrics, MetricsAggregator


def generate_text_summary(metrics: PreprocessingMetrics) -> str:
    """
    Generate a human-readable text summary of preprocessing metrics.

    Args:
        metrics: PreprocessingMetrics object

    Returns:
        Formatted text summary
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"PREPROCESSING QUALITY REPORT")
    lines.append("=" * 80)
    lines.append(f"File: {metrics.examination_id}")
    lines.append(f"Institution: {metrics.institution_id}")
    lines.append(f"Dataset: {metrics.data_group}")
    lines.append(f"Processing Time: {metrics.processing_timestamp}")
    lines.append(f"Duration: {metrics.processing_duration_seconds:.2f}s" if metrics.processing_duration_seconds else "Duration: N/A")
    lines.append("")

    lines.append("SIGNAL PROPERTIES:")
    lines.append(f"  Sampling Frequency: {metrics.sampling_frequency_hz} Hz")
    lines.append(f"  Recording Duration: {metrics.recording_duration_seconds:.2f}s")
    lines.append(f"  Channels: {metrics.n_channels}")
    lines.append("")

    if metrics.raw_signal_quality:
        sq = metrics.raw_signal_quality
        lines.append("RAW SIGNAL QUALITY:")
        lines.append(f"  Mean Amplitude: {sq.mean_amplitude_uv:.2f} µV")
        lines.append(f"  Std Amplitude: {sq.std_amplitude_uv:.2f} µV")
        lines.append(f"  Max Amplitude: {sq.max_amplitude_uv:.2f} µV")
        lines.append(f"  Median MAD: {sq.median_mad_uv:.2f} µV")
        lines.append("")

    if metrics.preprocessed_signal_quality:
        sq = metrics.preprocessed_signal_quality
        lines.append("PREPROCESSED SIGNAL QUALITY:")
        lines.append(f"  Mean Amplitude: {sq.mean_amplitude_uv:.2f} µV")
        lines.append(f"  Std Amplitude: {sq.std_amplitude_uv:.2f} µV")
        lines.append(f"  Max Amplitude: {sq.max_amplitude_uv:.2f} µV")
        lines.append(f"  Median MAD: {sq.median_mad_uv:.2f} µV")
        lines.append("")

    if metrics.filters_applied:
        lines.append("FILTERS APPLIED:")
        for filt in metrics.filters_applied:
            lines.append(f"  - {filt}")
        lines.append("")

    if metrics.segmentation:
        seg = metrics.segmentation
        lines.append("SEGMENTATION & QUALITY CONTROL:")
        lines.append(f"  Total Segments: {seg.total_segments}")
        lines.append(f"  Rejected (Amplitude): {seg.segments_rejected_amplitude}")
        lines.append(f"  Rejected (Flat): {seg.segments_rejected_flat}")
        lines.append(f"  Total Rejected: {seg.segments_rejected_total}")
        lines.append(f"  Segments Kept: {seg.segments_kept}")
        lines.append(f"  Rejection Rate: {seg.rejection_rate:.1%}")
        lines.append("")

    if metrics.warnings:
        lines.append("WARNINGS:")
        for warning in metrics.warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")

    lines.append(f"STATUS: {'✓ SUCCESS' if metrics.preprocessing_successful else '✗ FAILED'}")
    if metrics.failure_reason:
        lines.append(f"FAILURE REASON: {metrics.failure_reason}")

    lines.append("=" * 80)

    return "\n".join(lines)


def metrics_to_dataframe(metrics_list: List[PreprocessingMetrics]) -> pd.DataFrame:
    """
    Convert list of metrics to pandas DataFrame for analysis.

    Args:
        metrics_list: List of PreprocessingMetrics objects

    Returns:
        DataFrame with one row per file
    """
    rows = []
    for m in metrics_list:
        row = {
            'examination_id': m.examination_id,
            'institution_id': m.institution_id,
            'data_group': m.data_group,
            'processing_timestamp': m.processing_timestamp,
            'processing_duration_s': m.processing_duration_seconds,
            'sampling_freq_hz': m.sampling_frequency_hz,
            'recording_duration_s': m.recording_duration_seconds,
            'n_channels': m.n_channels,
            'preprocessing_successful': m.preprocessing_successful,
            'failure_reason': m.failure_reason,
            'n_warnings': len(m.warnings),
        }

        # Add raw signal quality
        if m.raw_signal_quality:
            row['raw_mean_amp_uv'] = m.raw_signal_quality.mean_amplitude_uv
            row['raw_std_amp_uv'] = m.raw_signal_quality.std_amplitude_uv
            row['raw_max_amp_uv'] = m.raw_signal_quality.max_amplitude_uv
            row['raw_median_mad_uv'] = m.raw_signal_quality.median_mad_uv

        # Add preprocessed signal quality
        if m.preprocessed_signal_quality:
            row['prep_mean_amp_uv'] = m.preprocessed_signal_quality.mean_amplitude_uv
            row['prep_std_amp_uv'] = m.preprocessed_signal_quality.std_amplitude_uv
            row['prep_max_amp_uv'] = m.preprocessed_signal_quality.max_amplitude_uv
            row['prep_median_mad_uv'] = m.preprocessed_signal_quality.median_mad_uv

        # Add segmentation metrics
        if m.segmentation:
            row['total_segments'] = m.segmentation.total_segments
            row['rejected_amplitude'] = m.segmentation.segments_rejected_amplitude
            row['rejected_flat'] = m.segmentation.segments_rejected_flat
            row['segments_kept'] = m.segmentation.segments_kept
            row['rejection_rate'] = m.segmentation.rejection_rate

        rows.append(row)

    return pd.DataFrame(rows)


def save_site_comparison_report(
    aggregator: MetricsAggregator,
    output_path: str
):
    """
    Save a site comparison report to CSV.

    Args:
        aggregator: MetricsAggregator with metrics from multiple files
        output_path: Path to save CSV file
    """
    summary = aggregator.get_all_sites_summary()

    rows = []
    for site_id, stats in summary.items():
        rows.append(stats)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Site comparison report saved to: {output_path}")


def print_site_comparison(aggregator: MetricsAggregator):
    """
    Print a formatted site comparison to console.

    Args:
        aggregator: MetricsAggregator with metrics from multiple files
    """
    summary = aggregator.get_all_sites_summary()

    print("\n" + "=" * 100)
    print("SITE COMPARISON REPORT")
    print("=" * 100)

    for site_id, stats in summary.items():
        print(f"\n{site_id}:")
        print(f"  Total Files: {stats['total_files']}")
        print(f"  Success Rate: {(stats['successful'] / stats['total_files'] * 100):.1f}%")

        if stats['rejection_rate_mean'] is not None:
            print(f"  Avg Rejection Rate: {stats['rejection_rate_mean']:.1%} ± {stats['rejection_rate_std']:.1%}")

        if stats['mean_amplitude_mean'] is not None:
            print(f"  Avg Signal Amplitude: {stats['mean_amplitude_mean']:.2f} ± {stats['mean_amplitude_std']:.2f} µV")

        print(f"  Total Warnings: {stats['total_warnings']}")

    print("=" * 100 + "\n")
