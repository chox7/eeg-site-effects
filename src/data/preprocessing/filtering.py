"""
EEG filtering functions.

This module contains functions for creating and applying various filters to EEG data.
"""

import warnings
import mne
from scipy.signal import iirnotch, BadCoefficients
from mne.filter import estimate_ringing_samples
from typing import List

from src.data.config.preprocessing_configs import FilterConfig


def create_filter(iir_params: dict, f_pass: float, f_stop: float, fs: float, btype: str) -> dict:
    """
    Create an IIR filter with given parameters.

    Args:
        iir_params: Filter parameters dictionary
        f_pass: Pass frequency for lowpass/highpass filters in Hz
        f_stop: Stop frequency for lowpass/highpass filters in Hz
        fs: Sampling frequency in Hz
        btype: Filter type ('highpass', 'lowpass')

    Returns:
        Filter parameters in SOS format

    Raises:
        BadCoefficients: If filter design fails
    """
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            iir_params = mne.filter.construct_iir_filter(
                iir_params,
                f_pass=f_pass,
                f_stop=f_stop,
                sfreq=fs,
                btype=btype,
                return_copy=False
            )
            return iir_params
    except BadCoefficients:
        raise BadCoefficients(f'Bad filter coefficients for {btype} filter')


def create_notch_filter(notch_freq: float, Q: float, fs: float) -> dict:
    """
    Create a notch filter for removing line noise.

    Args:
        notch_freq: Notch frequency in Hz (typically 50 Hz in EU, 60 Hz in US)
        Q: Quality factor (notch_freq / notch_widths)
        fs: Sampling frequency in Hz

    Returns:
        Filter parameters dictionary

    Raises:
        BadCoefficients: If filter design fails
    """
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)
            padlen = estimate_ringing_samples((b, a))
            iir_params = {'output': 'ba', 'b': b, 'a': a, 'padlen': padlen}
            return iir_params
    except BadCoefficients:
        raise BadCoefficients(f'Bad filter coefficients for notch filter')


def build_filters(fs: float, filter_configs: List[FilterConfig]) -> List[dict]:
    """
    Build a list of filters based on provided configurations.

    Args:
        fs: Sampling frequency in Hz (from edf.info['sfreq'])
        filter_configs: List of FilterConfig objects

    Returns:
        List of filter parameters in appropriate format

    Raises:
        ValueError: If unsupported filter type is specified
    """
    filters = []

    for config in filter_configs:
        if config.type == 'highpass':
            iir_params = dict(
                ftype=config.ftype,
                output='sos',
                gpass=config.gpass,
                gstop=config.gstop
            )
            iir_params = create_filter(
                iir_params,
                config.f_pass,
                config.f_stop,
                fs,
                'highpass'
            )
            filters.append(iir_params)

        elif config.type == 'lowpass':
            iir_params = dict(
                ftype=config.ftype,
                output='sos',
                gpass=config.gpass,
                gstop=config.gstop
            )
            iir_params = create_filter(
                iir_params,
                config.f_pass,
                config.f_stop,
                fs,
                'lowpass'
            )
            filters.append(iir_params)

        elif config.type == 'notch':
            Q = config.notch_freq / config.notch_widths
            iir_params = create_notch_filter(config.notch_freq, Q, fs)
            filters.append(iir_params)

        else:
            raise ValueError(f"Unsupported filter type: {config.type}")

    return filters


def apply_filters(edf: mne.io.Raw, filter_configs: List[FilterConfig]) -> mne.io.Raw:
    """
    Apply a sequence of filters to EEG data.

    Args:
        edf: MNE Raw object containing EEG data
        filter_configs: List of FilterConfig objects to apply sequentially

    Returns:
        Filtered MNE Raw object
    """
    fs = edf.info['sfreq']
    filters = build_filters(fs, filter_configs)

    for filt in filters:
        # l_freq and h_freq must be numbers, but actual filtering uses iir_params
        edf = edf.filter(iir_params=filt, l_freq=0, h_freq=0, method='iir')

    return edf
