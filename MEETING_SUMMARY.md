# Meeting Summary: EEG Site Effects Project

## Project Overview
**Research Question:** How do site effects (systematic non-biological variance from different medical centers) affect EEG data in multi-center studies?

**Dataset:** ELM19 - 26,568 normal EEG recordings from 30 hospitals

---

## What We've Done (Completed Work)

### 1. Data Preparation (notebooks/00_data_analysis/)
- Filtered institutions to those with ≥100 normal recordings
- Extracted 2,850 features per recording:
  - Coherence (2,394 features) - inter-electrode connectivity
  - Power (266 features) - band power per electrode
  - Covariance (190 features) - electrode pair covariance
- Explored demographics and recording parameters across sites

### 2. Quantified the Problem (notebooks/01_site_classification/)
**Key Result:** Sites are highly distinguishable from EEG features
- **MCC = 0.865, Accuracy = 87.8%, AUC = 0.992**
- This proves site effects are real and significant

**SHAP Analysis revealed:**
- Frontal regions (F7, Fp1, Fp2) most important
- Low frequencies (delta/theta) most discriminative
- Each hospital has unique feature signatures

**Ablation Study:**
- Site effects distributed across ALL feature types (coherence, power, covariance)
- Simple feature selection cannot remove them

### 3. Attempted Harmonization (notebooks/02_combat_harmonization/)
Tested standard batch correction methods:
- SiteWiseStandardScaler (Z-score per site)
- ComBat (Johnson's method)
- NeuroCombat (with covariates)
- CovBat (covariance harmonization)

**The Paradox:**
| Method | Site Classification MCC |
|--------|------------------------|
| Raw (baseline) | 0.865 |
| ComBat | **0.959** (worse!) |
| StandardScaler | ~0.97 (worse!) |
| CovBat | ~0.95 (worse!) |

**Interpretation:** Harmonization makes sites MORE distinguishable, not less!

### 4. Pathology Classification Impact
- Harmonization slightly improves pathology classification (AUC: 0.804 → 0.813)
- But this modest gain comes with dramatically increased site identifiability

### 5. PCA Sensitivity Analysis (notebooks/04_pca_sensitivity/)
- PCA transformation before harmonization **resolves the paradox**
- Key insight: It's not about dimensionality reduction, but the **transformation to an orthogonal basis**
- ComBat's per-feature corrections in correlated space create cross-feature artifacts
- In PCA space (decorrelated features), adjustments are "cleaner"
- At PCA 0.80: harmonized MCC drops to ~0.56 (vs 0.96 without PCA)
- Trade-off: ~3% pathology AUC cost (0.81 → 0.78)

### 6. Filter Robustness (notebooks/05_experiment_filters/)
- Tested 5 different filter configurations
- Results are **robust**: MCC range 0.857-0.867
- Validates that findings are not preprocessing artifacts

---

## Key Finding: The Harmonization Paradox

**Why this happens:**
1. Site effects are encoded in **correlations between features**, not just feature values
2. ComBat's per-feature location/scale corrections in correlated space create cross-feature artifacts
3. These artifacts expose deeper site-specific patterns

**Solution discovered:**
- PCA transformation before harmonization resolves the paradox
- The orthogonal (decorrelated) basis prevents cross-feature artifacts
- This explains why the issue is not about dimensionality but about **feature correlation structure**

---

## What's Left to Do (Future Work)

### 1. Domain-Adversarial Neural Network (DANN)
- **Goal:** Learn features that are discriminative for pathology but invariant to site
- **Architecture:** Neural network with gradient reversal layer
- Feature extractor → Task classifier (pathology)
                  → Domain classifier (site, with reversed gradients)

### 2. Alternative Approaches to Consider
- Different neural network architectures (CNN on raw EEG, transformers)
- Multi-task learning with site prediction as auxiliary task
- Domain adaptation / transfer learning methods
- Alternative feature engineering (wavelet, time-frequency)

### 3. Evaluation Framework
- Same LOGO (Leave-One-Site-Out) validation
- Compare: Site classification MCC + Pathology classification AUC
- Goal: Reduce site MCC while maintaining/improving pathology AUC

---

## For the New Team Member

**Key Concepts to Understand:**
1. **Site effects** = systematic differences between recording centers (not biological)
2. **The paradox** = standard harmonization makes sites easier to classify
3. **DANN** = adversarial approach to learn site-invariant features

**Important Files:**
- `src/data/processing_pipeline.py` - Feature extraction
- `src/harmonization/` - Current harmonization methods
- `experiments/ml/` - Classification experiments
- `notebooks/03_paradox_analysis/` - Core findings

**Getting Started:**
1. Read CLAUDE.md for project structure
2. Review notebooks 01-03 for background
3. Look at the paradox results in `results/tables/03_paradox_analysis/`

---

## Key Figures to Show

**All meeting-ready figures are in `paper/results/`:**

1. `main_ml_model/hospital_classification_avg_mcc_per_class_notitle.png` - Baseline
2. `harmonization_paradox/hospital_classification_mcc_harmonization.png` - **The paradox**
3. `harmonization_paradox/exp01_*_shap_overview.png` - SHAP comparison (5 methods)
4. `harmonization_paradox/cohen_d_harmonization_comparison.png` - Effect sizes
5. `shap_analysis_all_feats/mean_shap_eeg_feature_overview.png` - Feature importance
6. `data_analysis/` - Demographics (age, gender, duration)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total recordings | 26,568 (normal only) |
| Hospitals | 30 |
| Features | 2,850 |
| Site Classification MCC (raw) | 0.865 |
| Site Classification MCC (harmonized) | ~0.95 (paradox!) |
| Site Classification MCC (PCA + harmonized) | ~0.56 (resolved!) |
| Pathology Classification AUC | 0.80-0.81 |
