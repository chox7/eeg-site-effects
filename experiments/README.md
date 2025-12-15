# Experiments

This directory contains runnable experiment scripts.

## Structure

- **`preprocessing/`** - Preprocessing experiments (e.g., testing different filters)
- **`ml/`** - Machine learning experiments (training and evaluation)

## Usage

All scripts are meant to be run from the project root:

```bash
# Preprocessing experiments
python experiments/preprocessing/filter_comparison.py

# ML experiments
python experiments/ml/site_classification.py
python experiments/ml/pathology_classification.py
```

## Difference from `src/`

- **`src/`** contains library code (importable modules)
- **`experiments/`** contains scripts that USE the library to run experiments
