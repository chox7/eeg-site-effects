"""
Configuration classes for EEG preprocessing pipeline.

This module defines configuration classes for different preprocessing steps,
allowing for site-specific and experiment-specific customization.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FilterConfig:
    """Configuration for EEG filtering.

    Attributes:
        type: Filter type ('highpass', 'lowpass', 'notch')
        f_pass: Passband edge frequency in Hz (for highpass/lowpass)
        f_stop: Stopband edge frequency in Hz (for highpass/lowpass)
        notch_freq: Notch frequency in Hz (for notch filter)
        notch_widths: Notch width in Hz (for notch filter)
        gpass: Maximum loss in passband (dB)
        gstop: Minimum attenuation in stopband (dB)
        ftype: Filter type ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')
    """
    type: str
    f_pass: Optional[float] = None
    f_stop: Optional[float] = None
    notch_freq: Optional[float] = None
    notch_widths: Optional[float] = None
    gpass: float = 1.0
    gstop: float = 20.0
    ftype: str = 'butter'


@dataclass
class QualityControlConfig:
    """Configuration for quality control and artifact rejection.

    Attributes:
        crop_length_samples: Length of each signal segment in samples
        max_amplitude_uv: Maximum allowed amplitude in µV
        min_mad_uv: Minimum MAD threshold in µV for flat channel rejection
    """
    crop_length_samples: int = 600
    max_amplitude_uv: float = 800.0
    min_mad_uv: float = 1.0


@dataclass
class PreprocessingConfig:
    """Main preprocessing configuration.

    Note: Sampling frequency is extracted from each EDF file via edf.info['sfreq']
    as different hospitals may record at different sampling rates.

    Attributes:
        desired_sampling_freq: Target sampling frequency for feature extraction
        filters: List of filter configurations to apply sequentially
        quality_control: Quality control configuration
        apply_average_reference: Whether to apply average reference
        site_id: Optional site/institution identifier for site-specific processing
    """
    desired_sampling_freq: float = 100.0
    filters: List[FilterConfig] = field(default_factory=lambda: [
        FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
        FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ])
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    apply_average_reference: bool = True
    site_id: Optional[str] = None


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction.

    Attributes:
        extract_power: Whether to extract power features
        extract_connectivity: Whether to extract connectivity/coherence features
        extract_covariance: Whether to extract covariance features
        connectivity_method: Method for connectivity computation
        connectivity_mode: Mode for connectivity computation
        psd_method: Method for power spectral density
        normalize_power: Whether to normalize power features
        aggregation_method: Method to aggregate across segments
    """
    extract_power: bool = True
    extract_connectivity: bool = True
    extract_covariance: bool = True
    connectivity_method: str = 'coh'
    connectivity_mode: str = 'multitaper'
    psd_method: str = 'multitaper'
    normalize_power: bool = True
    aggregation_method: str = 'median'


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_FEATURE_CONFIG = FeatureExtractionConfig()


def get_site_specific_config(site_id: str) -> PreprocessingConfig:
    """
    Get site-specific preprocessing configuration.

    Args:
        site_id: Site/institution identifier

    Returns:
        PreprocessingConfig: Site-specific preprocessing configuration
    """
    base_config = PreprocessingConfig(site_id=site_id)

    if site_id == 'US_SITE':
        base_config.filters = [
            FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
            FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
            FilterConfig(type='notch', notch_freq=60.0, notch_widths=10.0)
        ]

    return base_config


def create_custom_config(
    highpass_freq: float = 0.5,
    lowpass_freq: float = 40.0,
    notch_freq: float = 50.0,
    max_amplitude_uv: float = 800.0,
    crop_length_samples: int = 600,
    **kwargs
) -> PreprocessingConfig:
    """
    Create a custom preprocessing configuration.

    Args:
        highpass_freq: High-pass filter cutoff frequency in Hz
        lowpass_freq: Low-pass filter cutoff frequency in Hz
        notch_freq: Notch filter frequency in Hz
        max_amplitude_uv: Maximum amplitude threshold in µV
        crop_length_samples: Segment length in samples
        **kwargs: Additional arguments passed to PreprocessingConfig

    Returns:
        PreprocessingConfig: Custom preprocessing configuration
    """
    filters = [
        FilterConfig(type='highpass', f_pass=highpass_freq, f_stop=highpass_freq - 0.4),
        FilterConfig(type='lowpass', f_pass=lowpass_freq, f_stop=lowpass_freq + 10.0),
        FilterConfig(type='notch', notch_freq=notch_freq, notch_widths=10.0)
    ]

    qc_config = QualityControlConfig(
        crop_length_samples=crop_length_samples,
        max_amplitude_uv=max_amplitude_uv
    )

    return PreprocessingConfig(
        filters=filters,
        quality_control=qc_config,
        **kwargs
    )
