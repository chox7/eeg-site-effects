"""
EEG constants for the 10-20 system and frequency band definitions.

This module consolidates all EEG-related constants used throughout the project.
"""

import numpy as np

# --- CHANNEL DEFINITIONS ---

# Standard 10-20 system channel names (19 channels)
CHANNELS_19 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# 2D channel positions for topographic plots (normalized coordinates)
CHANNEL_POSITIONS_19 = {
    "Fp1": (-0.5, 1.0),
    "Fp2": (0.5, 1.0),
    "F7": (-1.0, 0.5),
    "F3": (-0.5, 0.5),
    "Fz": (0.0, 0.5),
    "F4": (0.5, 0.5),
    "F8": (1.0, 0.5),
    "T3": (-1.2, 0.0),
    "C3": (-0.5, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0),
    "T4": (1.2, 0.0),
    "T5": (-1.0, -0.5),
    "P3": (-0.5, -0.5),
    "Pz": (0.0, -0.5),
    "P4": (0.5, -0.5),
    "T6": (1.0, -0.5),
    "O1": (-0.5, -1.0),
    "O2": (0.5, -1.0),
}


# --- FREQUENCY BAND DEFINITIONS ---

# 14 overlapping frequency bands used for feature extraction (Hz)
FREQ_BANDS_14 = np.array([
    [0.5, 2],    # delta 1
    [1, 3],      # delta 2
    [2, 4],      # delta 3
    [3, 6],      # theta 1
    [4, 8],      # theta 2
    [6, 10],     # alpha 1
    [8, 13],     # alpha 2
    [10, 15],    # beta 1
    [13, 18],    # beta 2
    [15, 21],    # beta 3
    [18, 24],    # beta 4
    [21, 27],    # beta 5
    [24, 30],    # gamma 1
    [27, 40],    # gamma 2
])

# Band names corresponding to FREQ_BANDS_14
FREQ_BAND_NAMES_14 = np.array([
    "delta", "delta", "delta",
    "theta", "theta",
    "alpha", "alpha",
    "beta", "beta", "beta", "beta", "beta",
    "gamma", "gamma",
])

# Traditional EEG frequency bands (grouped by clinical convention)
FREQ_BANDS_NAMED = {
    "delta": [[0.5, 2], [1, 3], [2, 4]],
    "theta": [[3, 6], [4, 8]],
    "alpha": [[6, 10], [8, 13]],
    "beta": [[10, 15], [13, 18], [15, 21], [18, 24], [21, 27]],
    "gamma": [[24, 30], [27, 40]],
}


# --- CHANNEL NAME MAPPINGS ---

# Mapping from various EDF channel naming conventions to standard names
CHANNEL_NAME_MAPPINGS = [
    # Standard European EEG format
    {
        "EEG Fp1": "Fp1", "EEG Fp2": "Fp2",
        "EEG F7": "F7", "EEG F3": "F3", "EEG Fz": "Fz", "EEG F4": "F4", "EEG F8": "F8",
        "EEG T3": "T3", "EEG C3": "C3", "EEG Cz": "Cz", "EEG C4": "C4", "EEG T4": "T4",
        "EEG T5": "T5", "EEG P3": "P3", "EEG Pz": "Pz", "EEG P4": "P4", "EEG T6": "T6",
        "EEG O1": "O1", "EEG O2": "O2",
    },
    # MOR hospital format (with Fz_nowe)
    {
        "EEG Fp1": "Fp1", "EEG Fp2": "Fp2",
        "EEG F7": "F7", "EEG F3": "F3", "Fz_nowe": "Fz", "EEG F4": "F4", "EEG F8": "F8",
        "EEG T3": "T3", "EEG C3": "C3", "EEG Cz": "Cz", "EEG C4": "C4", "EEG T4": "T4",
        "EEG T5": "T5", "EEG P3": "P3", "EEG Pz": "Pz", "EEG P4": "P4", "EEG T6": "T6",
        "EEG O1": "O1", "EEG O2": "O2",
    },
    # TUH format (with -REF suffix)
    {
        "EEG FP1-REF": "Fp1", "EEG FP2-REF": "Fp2",
        "EEG F7-REF": "F7", "EEG F3-REF": "F3", "EEG FZ-REF": "Fz",
        "EEG F4-REF": "F4", "EEG F8-REF": "F8",
        "EEG T3-REF": "T3", "EEG C3-REF": "C3", "EEG CZ-REF": "Cz",
        "EEG C4-REF": "C4", "EEG T4-REF": "T4",
        "EEG T5-REF": "T5", "EEG P3-REF": "P3", "EEG PZ-REF": "Pz",
        "EEG P4-REF": "P4", "EEG T6-REF": "T6",
        "EEG O1-REF": "O1", "EEG O2-REF": "O2",
        "EEG A1-REF": "A1", "EEG A2-REF": "A2",
    },
    # Alternative -REF format (lowercase start)
    {
        "Fp1-REF": "Fp1", "Fp2-REF": "Fp2",
        "F7-REF": "F7", "F3-REF": "F3", "Fz-REF": "Fz", "F4-REF": "F4", "F8-REF": "F8",
        "T3-REF": "T3", "C3-REF": "C3", "Cz-REF": "Cz", "C4-REF": "C4", "T4-REF": "T4",
        "T5-REF": "T5", "P3-REF": "P3", "Pz-REF": "Pz", "P4-REF": "P4", "T6-REF": "T6",
        "O1-REF": "O1", "O2-REF": "O2",
    },
]


# --- FEATURE COUNTS ---

# Expected feature counts for standard 19-channel setup with 14 bands
N_CHANNELS = len(CHANNELS_19)  # 19
N_CHANNEL_PAIRS = N_CHANNELS * (N_CHANNELS - 1) // 2  # 171 (upper triangle)
N_FREQ_BANDS = len(FREQ_BANDS_14)  # 14

N_COHERENCE_FEATURES = N_CHANNEL_PAIRS * N_FREQ_BANDS  # 2394
N_POWER_FEATURES = N_CHANNELS * N_FREQ_BANDS  # 266
N_COVARIANCE_FEATURES = N_CHANNELS * (N_CHANNELS + 1) // 2  # 190 (with diagonal)
N_TOTAL_FEATURES = N_COHERENCE_FEATURES + N_POWER_FEATURES + N_COVARIANCE_FEATURES  # 2850
