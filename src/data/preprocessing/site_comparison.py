"""
Site comparison visualization tools.

This module provides functions to compare preprocessing quality and signal
characteristics across different hospitals/sites.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

from src.data.preprocessing.quality_metrics import MetricsAggregator, PreprocessingMetrics


def plot_rejection_rates_by_site(
    aggregator: MetricsAggregator,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot rejection rates across different sites.

    Args:
        aggregator: MetricsAggregator with metrics from multiple sites
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    summary = aggregator.get_all_sites_summary()

    sites = []
    mean_rejection = []
    std_rejection = []

    for site_id, stats in summary.items():
        if stats['rejection_rate_mean'] is not None:
            sites.append(site_id)
            mean_rejection.append(stats['rejection_rate_mean'] * 100)
            std_rejection.append(stats['rejection_rate_std'] * 100)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(sites))
    bars = ax.bar(x, mean_rejection, yerr=std_rejection, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Site/Hospital')
    ax.set_ylabel('Rejection Rate (%)')
    ax.set_title('Segment Rejection Rates by Site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_rejection, std_rejection)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.1f}±{std_val:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_signal_quality_by_site(
    aggregator: MetricsAggregator,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare signal quality metrics across sites.

    Args:
        aggregator: MetricsAggregator with metrics from multiple sites
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    summary = aggregator.get_all_sites_summary()

    sites = []
    mean_amp_mean = []
    mean_amp_std = []

    for site_id, stats in summary.items():
        if stats['mean_amplitude_mean'] is not None:
            sites.append(site_id)
            mean_amp_mean.append(stats['mean_amplitude_mean'])
            mean_amp_std.append(stats['mean_amplitude_std'])

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(sites))
    bars = ax.bar(x, mean_amp_mean, yerr=mean_amp_std, capsize=5, alpha=0.7, color='coral')
    ax.set_xlabel('Site/Hospital')
    ax.set_ylabel('Mean Signal Amplitude (µV)')
    ax.set_title('Signal Amplitude by Site (After Preprocessing)')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean_val, std_val in zip(bars, mean_amp_mean, mean_amp_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.1f}±{std_val:.1f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_site_comparison_dashboard(
    aggregator: MetricsAggregator,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comprehensive dashboard comparing all sites.

    Args:
        aggregator: MetricsAggregator with metrics from multiple sites
        figsize: Figure size

    Returns:
        Matplotlib figure with multiple comparison plots
    """
    summary = aggregator.get_all_sites_summary()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Extract data
    sites = []
    total_files = []
    success_rates = []
    rejection_rates = []
    rejection_stds = []
    mean_amps = []
    amp_stds = []

    for site_id, stats in summary.items():
        sites.append(site_id)
        total_files.append(stats['total_files'])
        success_rate = (stats['successful'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
        success_rates.append(success_rate)

        if stats['rejection_rate_mean'] is not None:
            rejection_rates.append(stats['rejection_rate_mean'] * 100)
            rejection_stds.append(stats['rejection_rate_std'] * 100)
        else:
            rejection_rates.append(0)
            rejection_stds.append(0)

        if stats['mean_amplitude_mean'] is not None:
            mean_amps.append(stats['mean_amplitude_mean'])
            amp_stds.append(stats['mean_amplitude_std'])
        else:
            mean_amps.append(0)
            amp_stds.append(0)

    x = np.arange(len(sites))

    # 1. Success rates
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x, success_rates, alpha=0.7, color='green')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Preprocessing Success Rate by Site')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sites, rotation=45, ha='right')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. Rejection rates
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, rejection_rates, yerr=rejection_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_ylabel('Rejection Rate (%)')
    ax2.set_title('Segment Rejection Rate by Site')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Signal amplitudes
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x, mean_amps, yerr=amp_stds, capsize=5, alpha=0.7, color='steelblue')
    ax3.set_ylabel('Mean Amplitude (µV)')
    ax3.set_title('Signal Amplitude by Site')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sites, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Sample sizes
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(x, total_files, alpha=0.7, color='purple')
    ax4.set_ylabel('Number of Files')
    ax4.set_title('Sample Size by Site')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sites, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, total_files):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Multi-Site Preprocessing Comparison Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_metric_distribution_by_site(
    metrics_list: List[PreprocessingMetrics],
    metric_name: str = 'rejection_rate',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of a specific metric across sites using box plots.

    Args:
        metrics_list: List of PreprocessingMetrics objects
        metric_name: Name of metric to plot ('rejection_rate', 'mean_amplitude', etc.)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Group metrics by site
    site_data = {}
    for m in metrics_list:
        if not m.preprocessing_successful:
            continue

        site_id = m.institution_id
        if site_id not in site_data:
            site_data[site_id] = []

        # Extract metric value
        if metric_name == 'rejection_rate' and m.segmentation:
            site_data[site_id].append(m.segmentation.rejection_rate * 100)
        elif metric_name == 'mean_amplitude' and m.preprocessed_signal_quality:
            site_data[site_id].append(m.preprocessed_signal_quality.mean_amplitude_uv)
        elif metric_name == 'max_amplitude' and m.preprocessed_signal_quality:
            site_data[site_id].append(m.preprocessed_signal_quality.max_amplitude_uv)
        elif metric_name == 'median_mad' and m.preprocessed_signal_quality:
            site_data[site_id].append(m.preprocessed_signal_quality.median_mad_uv)

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    sites = list(site_data.keys())
    data = [site_data[site] for site in sites]

    bp = ax.boxplot(data, labels=sites, patch_artist=True)

    # Customize colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('Site/Hospital')
    if metric_name == 'rejection_rate':
        ax.set_ylabel('Rejection Rate (%)')
        ax.set_title('Distribution of Rejection Rates by Site')
    elif metric_name == 'mean_amplitude':
        ax.set_ylabel('Mean Amplitude (µV)')
        ax.set_title('Distribution of Signal Amplitude by Site')
    else:
        ax.set_ylabel(metric_name)
        ax.set_title(f'Distribution of {metric_name} by Site')

    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig
