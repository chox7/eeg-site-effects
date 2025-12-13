"""
Channel operations for EEG preprocessing.

This module handles channel standardization, reordering, and rereferencing.
"""

import mne
import numpy as np
from src.utils.utils import CHNAMES_MAPPING, CH_NAMES


class ChannelsError(Exception):
    """Exception raised for errors in channel renaming or reordering."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def standardize_electrode_names_and_order(edf: mne.io.Raw) -> mne.io.Raw:
    """
    Standardize electrode names and reorder channels to match CH_NAMES.

    This function tries multiple channel name mappings from CHNAMES_MAPPING
    until one succeeds. Different hospitals may use different naming conventions.

    Args:
        edf: MNE Raw object with potentially non-standard channel names

    Returns:
        MNE Raw object with standardized channel names and order

    Raises:
        ChannelsError: If channel renaming or reordering fails for all mappings
    """
    for mapping in CHNAMES_MAPPING:
        try:
            edf = edf.rename_channels(mapping)
            edf = edf.reorder_channels(CH_NAMES)
            return edf
        except ValueError:
            continue

    raise ChannelsError(
        f"Channel renaming or reordering failed. Available channels: {edf.info['ch_names']}. "
        f"Required channels: {CH_NAMES}."
    )


def apply_average_reference(edf: mne.io.Raw) -> mne.io.Raw:
    """
    Apply average reference to EEG data.

    Average reference is computed as: -sum(all_channels) / (n_channels + 1)
    This is then added to each channel.

    Args:
        edf: MNE Raw object

    Returns:
        MNE Raw object with average reference applied
    """
    reference = -np.sum(edf._data, axis=0) / (edf._data.shape[0] + 1)
    edf._data += reference
    return edf


def set_standard_montage(edf: mne.io.Raw) -> mne.io.Raw:
    """
    Set standard 10-20 montage for EEG channels.

    Args:
        edf: MNE Raw object with standardized channel names

    Returns:
        MNE Raw object with montage set
    """
    edf = edf.set_montage("standard_1020", match_case=False)
    return edf
