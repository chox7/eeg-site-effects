# Experiments

This directory contains runnable experiment scripts.

## Structure

- **`preprocessing/`** - Preprocessing experiments (e.g., testing different filters)
- **`ml/`** - Machine learning experiments (training and evaluation)
- **`configs/`** - YAML configuration files for ML experiments

## Usage

All scripts are meant to be run from the project root.

### ML Experiments with Config Files (Recommended)

```bash
# Site classification - run all methods from config
python experiments/ml/site_classification.py --config experiments/configs/site_classification/default.yaml

# Site classification - run specific method
python experiments/ml/site_classification.py -c experiments/configs/site_classification/default.yaml -m combat

# Pathology classification - run all methods from config
python experiments/ml/pathology_classification.py --config experiments/configs/pathology_classification/default.yaml

# Pathology classification - run specific method
python experiments/ml/pathology_classification.py -c experiments/configs/pathology_classification/default.yaml -m raw
```

### Creating Custom Configs

Copy an existing config and modify paths/parameters:

```bash
cp experiments/configs/site_classification/default.yaml experiments/configs/site_classification/my_experiment.yaml
# Edit my_experiment.yaml with your paths and parameters
python experiments/ml/site_classification.py -c experiments/configs/site_classification/my_experiment.yaml
```

### Legacy Mode (Backward Compatible)

```bash
# Still works - uses default paths
python experiments/ml/site_classification.py raw
python experiments/ml/pathology_classification.py combat
```

### Preprocessing Experiments

```bash
python experiments/preprocessing/filter_comparison.py
```

## Config File Structure

YAML configs specify:
- **paths**: Input data files, output results/models/SHAP data directories
- **harmonization_methods**: List of methods to run (raw, sitewise, combat, neurocombat, covbat)
- **catboost_params**: ML hyperparameters
- **cv**: Cross-validation settings (n_splits, random_state, k_calibration)

See `experiments/configs/*/default.yaml` for examples.

## Difference from `src/`

- **`src/`** contains library code (importable modules)
- **`experiments/`** contains scripts that USE the library to run experiments
