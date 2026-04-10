# Experiments

This directory contains runnable experiment scripts and their YAML configurations.

## Structure

- **`preprocessing/`** — Data preparation scripts (DANN conversion, hospital anonymization, filter comparison)
- **`ml/`** — Machine learning experiment scripts (training and evaluation)
- **`configs/`** — YAML configuration files, one per experiment

## Configs

Each config maps to a results section:

| Config file | Section | Task |
|---|---|---|
| `site_classification.yaml` | 02_site_effect | Site classification (5-fold stratified CV) |
| `pathology_classification.yaml` | 03_harmonization | Pathology classification (LOSO CV + calibration) |
| `dann_site_classification.yaml` | 04_dann | Site clf on DANN-extracted features |
| `dann_pathology_classification.yaml` | 04_dann | Pathology clf on DANN-extracted features |

PCA sensitivity experiments (`05_pca_sensitivity`) are run via `ml/pca_sensitivity.py` with inline config.

## Usage

All scripts are run from the **project root** using the project venv:

```bash
# Site classification — all methods from config, CatBoost
.venv/bin/python experiments/ml/site_classification.py \
    -c experiments/configs/site_classification.yaml

# Pathology classification — single method, LogReg
.venv/bin/python experiments/ml/pathology_classification.py \
    -c experiments/configs/pathology_classification.yaml \
    --method combat --model logreg

# DANN pathology classification
.venv/bin/python experiments/ml/pathology_classification.py \
    -c experiments/configs/dann_pathology_classification.yaml

# With nohup (for long runs)
nohup .venv/bin/python experiments/ml/site_classification.py \
    -c experiments/configs/site_classification.yaml \
    --model catboost > nohup_site.out 2>&1 &
```

### CLI Flags (all ml scripts)

| Flag | Description | Default |
|---|---|---|
| `--config`, `-c` | YAML config file path | required |
| `--method`, `-m` | Override harmonization method | all from config |
| `--model` | `catboost` or `logreg` | `catboost` |
| `--tag`, `-t` | Tag for this run in results CSV | none |
| `--features`, `-f` | Override features CSV path | from config |
| `--info`, `-i` | Override info CSV path | from config |

## Difference from `src/`

- **`src/`** contains library code (importable modules)
- **`experiments/`** contains scripts that USE the library to run experiments
