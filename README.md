# Investigating Site Effects in Multi-Center EEG Data: A Machine Learning Approach

### Bachelor's Thesis in Neuroinformatics

This repository contains the source code and analysis for my thesis. The project tackles a critical challenge in medical machine learning: the "site effect," which is the systematic, non-biological variance that arises when combining data from different medical centers. This effect can severely degrade the performance and generalizability of diagnostic models.

**The complete thesis is available [here](paper/Investigating_Site_Effects_in_Multi_Center_EEG_Data_for_Neuroscreening__A_Machine_Learning_Approach.pdf).** 

---

### Project Goals

The goal was to quantify, interpret, and attempt to reduce the site effect in the ELM19 EEG dataset.

1.  **Quantification & Interpretation:**
    * A **CatBoost classifier** was trained to identify the source hospital from EEG features. The model's high accuracy (MCC = 0.843) confirmed a strong site effect.
    * **SHAP (SHapley Additive exPlanations)** analysis was used to interpret the classifier, revealing which EEG features were the biggest predictors of the source.

2.  **Harmonization & Evaluation:**
    * Standard data harmonization techniques (**ComBat**, site-wise standardization) were applied to the features to reduce inter-site differences.
    * The impact of harmonization was evaluated on two fronts: its ability to mask the source hospital and its effect on a downstream clinical neuroscreening task.

---

### Key Findings

- **The site effect is strong and easily detectable** by modern classifiers.
- **Standard harmonization methods are not a enough.** While they reduced feature-level differences, they paradoxically made the source hospitals even easier to classify (MCC 0.924--0.969), indicating that they might introduce new, subtle biases.
- Despite this, harmonization provided a **slight performance improvement** in the actual clinical task, highlighting the complex, non-linear nature of the problem.
---

### Project Structure

```
eeg-site-effects/
├── data/ELM19/                        # Source data (gitignored)
│   ├── filtered/                      # Preprocessed features + info CSVs
│   ├── dann/                          # DANN-extracted features (NPZ + CSV)
│   └── hospital_anonymization_mapping.csv
│
├── experiments/
│   ├── configs/                       # YAML configs for each experiment
│   │   ├── site_classification.yaml                # 02_site_effect
│   │   ├── pathology_classification.yaml           # 03_harmonization
│   │   ├── dann_site_classification.yaml           # 04_dann (final checkpoint)
│   │   ├── dann_pathology_classification.yaml      # 04_dann (final checkpoint)
│   │   ├── dann_best_site_classification.yaml      # 04_dann (best checkpoint)
│   │   └── dann_best_pathology_classification.yaml # 04_dann (best checkpoint)
│   ├── ml/                            # Runnable experiment scripts
│   │   ├── site_classification.py
│   │   ├── pathology_classification.py
│   │   └── pca_sensitivity.py
│   └── preprocessing/                 # Data preparation scripts
│
├── notebooks/                         # Analysis & visualization (one folder per section)
│   ├── 01_exploratory_analysis/
│   ├── 02_site_effect/
│   ├── 03_harmonization/
│   ├── 04_dann/
│   └── 05_pca_sensitivity/
│
├── results/
│   ├── tables/                        # CSV results (see below)
│   ├── figures/                       # Saved plots (PNG)
│   └── shap_data/                     # SHAP values for post-hoc analysis
│
├── src/                               # Importable library code
│   ├── config/                        # Experiment config dataclasses + YAML loading
│   ├── harmonization/                 # SiteWiseScaler (ComBat via combatlearn)
│   └── visualization/                 # SHAP utilities, plotting helpers
│
└── models/                            # Saved pipelines (gitignored)
```

#### Results folder — canonical structure

Each numbered folder maps to a section in the thesis. All results (tables, figures, SHAP data) use the **same numbering**.

```
results/tables/
├── 01_exploratory_analysis/       # §4.1 — Dataset statistics, distributions
├── 02_site_effect/                # §4.2–4.4 — Site classification
│   ├── site_clf_results.csv       #   Canonical: ALL site clf results
│   └── feature_ablation/          #   SHAP & feature group analysis
├── 03_harmonization/              # §4.5–4.6 — Pathology classification (LOSO)
│   └── patho_clf_results.csv      #   Canonical: ALL pathology clf results
├── 04_dann/                       # §4.7 — DANN feature evaluation
│   ├── site_clf_results.csv
│   └── patho_clf_results.csv
└── 05_pca_sensitivity/            # §4.8 — PCA variance threshold analysis
    ├── site_clf_pca_results.csv
    └── patho_clf_pca_results.csv
```

The same numbering applies to `results/figures/`, `results/shap_data/`, and `notebooks/`.

---

### Contributor Guidelines

#### 1. Every notebook must map to a paper section

Notebooks live in `notebooks/XX_section_name/` where `XX` matches the results folder number. If you add a new analysis, assign it the next available number and create matching folders in `results/tables/`, `results/figures/`, and `notebooks/`.

#### 2. All outputs must be saved to files

Every figure and table that a notebook produces **must** be saved to `results/figures/XX_*/` and `results/tables/XX_*/` respectively. The notebook is a means of generating the result — the saved file is the artifact. The `01_exploratory_analysis/` section is a good blueprint: the notebook saves every plot as PNG and every summary as CSV.

#### 3. Hospital IDs are anonymized

Source data uses anonymized hospital IDs (H1–H30). The mapping is in `data/ELM19/hospital_anonymization_mapping.csv`. Never commit real hospital names. If you add new info CSVs, anonymize them first:
```bash
.venv/bin/python experiments/preprocessing/anonymize_hospital_ids.py \
    --files data/ELM19/path/to/new_info.csv
```

#### 4. Experiment scripts & configs

Experiments are run via `experiments/ml/*.py` with YAML configs from `experiments/configs/`. Key flags supported by all scripts:

| Flag | Description |
|---|---|
| `--config`, `-c` | Path to YAML config file |
| `--method`, `-m` | Single harmonization method (overrides config) |
| `--model` | `catboost` (default) or `logreg` |
| `--tag`, `-t` | Free-form tag to identify a run in results |

Canonical CatBoost parameters (defined in configs, must stay consistent):
- `iterations=1500, learning_rate=0.07, depth=5, l2_leaf_reg=9, task_type=GPU, max_bin=32`

Example — run pathology classification with ComBat + LogReg:
```bash
.venv/bin/python experiments/ml/pathology_classification.py \
    -c experiments/configs/pathology_classification.yaml \
    --method combat --model logreg --tag my_experiment
```

#### 5. Result CSV format

All canonical result CSVs should include these columns where applicable:

| Column | Values | Description |
|---|---|---|
| `model` | catboost, logreg | Classifier used |
| `method` | raw, sitewise, combat, neurocombat, covbat | Harmonization method |
| `pca_var` | none, all, 0.99, 0.95, 0.90, 0.80 | PCA variance threshold |
| `scaler` | none, robust | Preprocessing scaler |
| `features_type` | manual, dann_final, dann_best | Feature extraction method |
| `tag` | free-form | Run identifier |

Results append to CSVs (`mode='a'`). To rerun cleanly, delete the target CSV first.

#### 6. Adding a new experiment section

1. Pick the next number: `XX_section_name`
2. Create: `notebooks/XX_section_name/`, `results/tables/XX_section_name/`, `results/figures/XX_section_name/`
3. Create a YAML config in `experiments/configs/` pointing to the new results path
4. In your notebook, save every figure with `plt.savefig(...)` and every table with `df.to_csv(...)`

---

### Technologies Used
- **Python**
- **Scikit-learn** & **CatBoost**
- **Pandas** & **NumPy**
- **MNE-Python**
- **SHAP**
