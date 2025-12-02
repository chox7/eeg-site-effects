# Investigating Site Effects in Multi-Center EEG Data: A Machine Learning Approach

### Bachelor's Thesis in Neuroinformatics

This repository contains the source code and analysis for my thesis. The project tackles a critical challenge in medical machine learning: the "site effect," which is the systematic, non-biological variance that arises when combining data from different medical centers. This effect can severely degrade the performance and generalizability of diagnostic models.

**The complete thesis is available [here](paper/Investigating_Site_Effects_in_Multi_Center_EEG_Data_for_Neuroscreening__A_Machine_Learning_Approach.pdf).** 

---

### Project Goals

The goal was to quantify, interpret, and attempt to reduce the site effect in the ELM19 EEG dataset.

1.  **Quantification & Interpretation:**
    * A **CatBoost classifier** was trained to identify the source hospital from EEG features. The model's high accuracy (MCC = 0.865) confirmed a strong site effect.
    * **SHAP (SHapley Additive exPlanations)** analysis was used to interpret the classifier, revealing which EEG features were the biggest predictors of the source.

2.  **Harmonization & Evaluation:**
    * Standard data harmonization techniques (**ComBat**, site-wise standardization) were applied to the features to reduce inter-site differences.
    * The impact of harmonization was evaluated on two fronts: its ability to mask the source hospital and its effect on a downstream clinical neuroscreening task.

---

### Key Findings

- **The site effect is strong and easily detectable** by modern classifiers.
- **Standard harmonization methods are not a enough.** While they reduced feature-level differences, they paradoxalnie made the source hospitals even easier to classify (MCC > 0.98), indicating that they might introduce new, subtle biases.
- Despite this, harmonization provided a **slight performance improvement** in the actual clinical task, highlighting the complex, non-linear nature of the problem.
---

### Technologies Used
- **Python**
- **Scikit-learn**
- **Pandas** & **NumPy**
- **MNE-Python**
