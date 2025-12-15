# Experiments

This directory contains runnable experiment scripts.

## Structure

- **`preprocessing/`** - Preprocessing experiments (e.g., testing different filters)
- **`ml/`** - Machine learning experiments (training and evaluation)
- **`configs/`** - YAML configuration files organized by experiment

## Config Organization

Configs follow the `XX_experiment_name` convention matching `results/`, `models/`, `notebooks/`:

```
configs/
└── 05_experiment_filters/
    ├── site_classification.yaml
    └── pathology_classification.yaml
```

## Usage

All scripts are meant to be run from the project root.

### ML Experiments with Config Files

```bash
# Site classification
python experiments/ml/site_classification.py -c experiments/configs/05_experiment_filters/site_classification.yaml

# Run specific method only
python experiments/ml/site_classification.py -c experiments/configs/05_experiment_filters/site_classification.yaml -m combat

# Pathology classification
python experiments/ml/pathology_classification.py -c experiments/configs/05_experiment_filters/pathology_classification.yaml
```

### Creating New Experiment Configs

```bash
mkdir experiments/configs/06_my_new_experiment
cp experiments/configs/05_experiment_filters/site_classification.yaml experiments/configs/06_my_new_experiment/
# Edit paths and parameters
```

### Preprocessing Experiments

```bash
python experiments/preprocessing/filter_comparison.py
```

## Difference from `src/`

- **`src/`** contains library code (importable modules)
- **`experiments/`** contains scripts that USE the library to run experiments
