"""
Visualization tools for preprocessing analysis.

This module provides functions to visualize preprocessing effects and
compare signals across different sites/configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Optional, List, Tuple
from scipy import signal

from src.data.preprocessing.quality_metrics import PreprocessingMetrics


def plot_signal_comparison(
    raw_edf: mne.io.Raw,
    preprocessed_edf: mne.io.Raw,
    channels: Optional[List[str]] = None,
    duration: float = 10.0,
    start_time: float = 0.0,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot raw vs preprocessed signals for visual comparison.

    Args:
        raw_edf: Raw EEG data before preprocessing
        preprocessed_edf: EEG data after preprocessing
        channels: List of channel names to plot (if None, plots first 4 channels)
        duration: Duration to plot in seconds
        start_time: Start time in seconds
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    if channels is None:
        channels = raw_edf.ch_names[:4]

    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 2, figsize=figsize)

    if n_channels == 1:
        axes = axes.reshape(1, -1)

    # Get data
    raw_data = raw_edf.get_data(picks=channels, units="uV")
    prep_data = preprocessed_edf.get_data(picks=channels, units="uV")

    # Time vector
    fs = raw_edf.info['sfreq']
    start_sample = int(start_time * fs)
    end_sample = int((start_time + duration) * fs)
    time = np.arange(start_sample, end_sample) / fs

    for i, ch_name in enumerate(channels):
        # Raw signal
        axes[i, 0].plot(time, raw_data[i, start_sample:end_sample], 'b-', linewidth=0.5)
        axes[i, 0].set_ylabel(f'{ch_name} (µV)')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('Raw Signal')
        if i == n_channels - 1:
            axes[i, 0].set_xlabel('Time (s)')

        # Preprocessed signal
        axes[i, 1].plot(time, prep_data[i, start_sample:end_sample], 'r-', linewidth=0.5)
        axes[i, 1].set_ylabel(f'{ch_name} (µV)')
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('Preprocessed Signal')
        if i == n_channels - 1:
            axes[i, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    return fig


def plot_power_spectrum_comparison(
    raw_edf: mne.io.Raw,
    preprocessed_edf: mne.io.Raw,
    channel: str,
    fmax: float = 50.0,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Compare power spectral density before and after preprocessing.

    Args:
        raw_edf: Raw EEG data
        preprocessed_edf: Preprocessed EEG data
        channel: Channel name to analyze
        fmax: Maximum frequency to display in Hz
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get data for specified channel
    raw_data = raw_edf.get_data(picks=[channel], units="uV").flatten()
    prep_data = preprocessed_edf.get_data(picks=[channel], units="uV").flatten()

    fs = raw_edf.info['sfreq']

    # Compute PSD using Welch's method
    freqs_raw, psd_raw = signal.welch(raw_data, fs=fs, nperseg=min(2048, len(raw_data)//2))
    freqs_prep, psd_prep = signal.welch(prep_data, fs=fs, nperseg=min(2048, len(prep_data)//2))

    # Plot raw PSD
    axes[0].semilogy(freqs_raw, psd_raw, 'b-', linewidth=1)
    axes[0].set_xlim([0, fmax])
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density (µV²/Hz)')
    axes[0].set_title(f'Raw Signal - {channel}')
    axes[0].grid(True, alpha=0.3)

    # Highlight frequency bands
    axes[0].axvspan(0.5, 4, alpha=0.1, color='purple', label='Delta')
    axes[0].axvspan(4, 8, alpha=0.1, color='blue', label='Theta')
    axes[0].axvspan(8, 13, alpha=0.1, color='green', label='Alpha')
    axes[0].axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
    axes[0].legend(loc='upper right', fontsize=8)

    # Plot preprocessed PSD
    axes[1].semilogy(freqs_prep, psd_prep, 'r-', linewidth=1)
    axes[1].set_xlim([0, fmax])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (µV²/Hz)')
    axes[1].set_title(f'Preprocessed Signal - {channel}')
    axes[1].grid(True, alpha=0.3)

    # Highlight frequency bands
    axes[1].axvspan(0.5, 4, alpha=0.1, color='purple')
    axes[1].axvspan(4, 8, alpha=0.1, color='blue')
    axes[1].axvspan(8, 13, alpha=0.1, color='green')
    axes[1].axvspan(13, 30, alpha=0.1, color='orange')

    plt.tight_layout()
    return fig


def plot_preprocessing_summary(
    raw_edf: mne.io.Raw,
    preprocessed_edf: mne.io.Raw,
    metrics: PreprocessingMetrics,
    channel: str = 'Cz',
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive preprocessing summary plot.

    Args:
        raw_edf: Raw EEG data
        preprocessed_edf: Preprocessed EEG data
        metrics: PreprocessingMetrics object
        channel: Channel to analyze
        figsize: Figure size

    Returns:
        Matplotlib figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Time series comparison
    ax1 = fig.add_subplot(gs[0, :])
    raw_data = raw_edf.get_data(picks=[channel], units="uV").flatten()
    prep_data = preprocessed_edf.get_data(picks=[channel], units="uV").flatten()
    fs = raw_edf.info['sfreq']
    time = np.arange(len(raw_data)) / fs

    # Plot first 10 seconds
    duration_samples = int(10 * fs)
    ax1.plot(time[:duration_samples], raw_data[:duration_samples], 'b-', alpha=0.5, label='Raw', linewidth=0.5)
    ax1.plot(time[:duration_samples], prep_data[:duration_samples], 'r-', alpha=0.7, label='Preprocessed', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'{channel} Amplitude (µV)')
    ax1.set_title(f'Signal Comparison - {channel}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. PSD comparison
    ax2 = fig.add_subplot(gs[1, 0])
    freqs_raw, psd_raw = signal.welch(raw_data, fs=fs, nperseg=min(2048, len(raw_data)//2))
    freqs_prep, psd_prep = signal.welch(prep_data, fs=fs, nperseg=min(2048, len(prep_data)//2))
    ax2.semilogy(freqs_raw, psd_raw, 'b-', alpha=0.7, label='Raw', linewidth=1)
    ax2.semilogy(freqs_prep, psd_prep, 'r-', alpha=0.7, label='Preprocessed', linewidth=1)
    ax2.set_xlim([0, 50])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (µV²/Hz)')
    ax2.set_title('Power Spectral Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Signal quality metrics
    ax3 = fig.add_subplot(gs[1, 1])
    metrics_names = ['Mean\nAmp', 'Std\nAmp', 'Max\nAmp', 'Median\nMAD']
    raw_metrics = [
        metrics.raw_signal_quality.mean_amplitude_uv,
        metrics.raw_signal_quality.std_amplitude_uv,
        metrics.raw_signal_quality.max_amplitude_uv,
        metrics.raw_signal_quality.median_mad_uv
    ]
    prep_metrics = [
        metrics.preprocessed_signal_quality.mean_amplitude_uv,
        metrics.preprocessed_signal_quality.std_amplitude_uv,
        metrics.preprocessed_signal_quality.max_amplitude_uv,
        metrics.preprocessed_signal_quality.median_mad_uv
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    ax3.bar(x - width/2, raw_metrics, width, label='Raw', alpha=0.7)
    ax3.bar(x + width/2, prep_metrics, width, label='Preprocessed', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylabel('Amplitude (µV)')
    ax3.set_title('Signal Quality Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Segmentation info
    ax4 = fig.add_subplot(gs[2, :])
    seg = metrics.segmentation
    categories = ['Total', 'Rejected\n(Amplitude)', 'Rejected\n(Flat)', 'Kept']
    counts = [seg.total_segments, seg.segments_rejected_amplitude,
              seg.segments_rejected_flat, seg.segments_kept]
    colors = ['gray', 'orange', 'red', 'green']

    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Number of Segments')
    ax4.set_title(f'Quality Control Summary - Rejection Rate: {seg.rejection_rate:.1%}')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    # Add metadata text
    info_text = (
        f"Institution: {metrics.institution_id}\n"
        f"Exam ID: {metrics.examination_id}\n"
        f"Sampling Freq: {metrics.sampling_frequency_hz} Hz\n"
        f"Duration: {metrics.recording_duration_seconds:.1f}s\n"
        f"Processing Time: {metrics.processing_duration_seconds:.2f}s"
    )
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


def save_preprocessing_report(
    raw_edf: mne.io.Raw,
    preprocessed_edf: mne.io.Raw,
    metrics: PreprocessingMetrics,
    output_path: str,
    channel: str = 'Cz'
):
    """
    Generate and save a comprehensive preprocessing report as PNG.

    Args:
        raw_edf: Raw EEG data
        preprocessed_edf: Preprocessed EEG data
        metrics: PreprocessingMetrics object
        output_path: Path to save the figure
        channel: Channel to analyze
    """
    fig = plot_preprocessing_summary(raw_edf, preprocessed_edf, metrics, channel)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Preprocessing report saved to: {output_path}")
