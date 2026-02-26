#!/bin/bash

# Run site and pathology classification for each filter configuration
# This script compares different EEG filter settings using raw (no harmonization) method
# Note: Run `pip install -e .` from project root first

set -e  # Exit on error

# Base paths
DATA_BASE="data/ELM19/experiments"
CONFIG_BASE="experiments/configs/05_experiment_filters"

# Filter configurations
FILTER_CONFIGS=(
    "original"
    "remove_low_freq_artifacts"
    "remove_high_freq_artifacts"
    "strict_both_ends"
    "clinical_range"
)

echo "========================================="
echo "Filter Comparison Experiment"
echo "========================================="
echo ""

for config in "${FILTER_CONFIGS[@]}"; do

    tag="${config}"
    info_path="${DATA_BASE}/${config}/info.csv"
    features_path="${DATA_BASE}/${config}/features.csv"
    features_norm_path="${DATA_BASE}/${config}/features_norm.csv"

    echo "-----------------------------------------"
    echo "Running: ${config}"
    echo "Info: ${info_path}"
    echo "Features for pathology classification: ${features_path}"
    echo "Features for site classification: ${features_norm_path}"
    echo "Tag: ${tag}"
    echo "-----------------------------------------"

    # Site Classification (uses normalized features)
    echo ">>> Site Classification..."
    python experiments/ml/site_classification.py \
        -c "${CONFIG_BASE}/site_classification.yaml" \
        -m raw \
        -f "${features_norm_path}" \
        -t "${tag}"

    # Pathology Classification (uses raw features)
    echo ">>> Pathology Classification..."
    python experiments/ml/pathology_classification.py \
        -c "${CONFIG_BASE}/pathology_classification.yaml" \
        -m raw \
        -i "${info_path}" \
        -f "${features_path}" \
        -t "${tag}"

    echo ""
done

echo "========================================="
echo "Filter Comparison Complete!"
echo "========================================="
