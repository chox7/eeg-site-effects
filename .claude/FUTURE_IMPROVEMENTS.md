# Future Improvements for Feature Extraction Pipeline

This document tracks potential improvements to the refactored feature extraction pipeline, prioritized by research value.

---

## ✅ COMPLETED

- [x] Modularize the pipeline into clean, testable components
- [x] Add configuration management for site-specific preprocessing
- [x] Implement comprehensive quality metrics collection
- [x] Add quality report generation (text, CSV, DataFrame)
- [x] Create preprocessing visualization tools
- [x] Add site comparison visualization dashboards
- [x] Create usage examples and documentation

---

## 🔴 HIGH PRIORITY - Most Valuable for Site Effects Research

### 1. Add Resampling Support
**Why:** Different hospitals may record at different sampling rates, which creates site effects.

**Implementation:**
- Add `target_sampling_freq` parameter that actually resamples
- Track original vs resampled frequency in metrics
- Compare site effects before/after standardizing sampling rates

**Files to modify:**
- `src/data/preprocessing/pipeline.py` - Add resampling step
- `src/data/config/preprocessing_configs.py` - Add resampling config

---

### 2. Test Different QC Thresholds Per Site
**Why:** Some hospitals might have inherently noisier data requiring different thresholds.

**Implementation:**
- Allow site-specific `max_amplitude_uv` and `min_mad_uv` in config
- Experiment with adaptive thresholds based on signal statistics
- Compare if site-specific QC reduces site effects

**Files to modify:**
- `src/data/config/preprocessing_configs.py` - Extend `get_site_specific_config()`

---

### 3. Spectral Feature Importance Analysis
**Why:** Identify which frequency bands contribute most to site differences.

**Implementation:**
- Add function to compute feature importance for site classification
- Identify problematic frequency bands per site
- Suggest targeted filtering based on analysis

**New files:**
- `src/data/analysis/spectral_analysis.py`
- Functions to compute band-wise site separability

---

### 4. Automatic Bad Channel Detection
**Why:** Consistently bad channels in certain hospitals create site effects.

**Implementation:**
- Detect channels with persistent artifacts (flat, excessive noise)
- Track bad channel statistics per site
- Option to interpolate or exclude bad channels

**Files to modify:**
- `src/data/preprocessing/quality_control.py` - Add bad channel detection
- `src/data/preprocessing/quality_metrics.py` - Track bad channel stats

---

### 5. Temporal Stability Checks
**Why:** Non-stationary signals indicate poor recording quality or artifacts.

**Implementation:**
- Compute signal stationarity metrics (Augmented Dickey-Fuller test)
- Flag files with strong temporal trends
- Track stationarity per site

**New files:**
- `src/data/preprocessing/stationarity.py`

---

## 🟡 MEDIUM PRIORITY - Robustness & Usability

### 6. Better Error Recovery
**Why:** Large batch processing shouldn't fail completely on single file errors.

**Implementation:**
- Add try-except around individual file processing
- Log errors with context (file, site, error type)
- Continue processing remaining files
- Generate error summary report

**Files to modify:**
- `src/data/feature_extraction.py` - Wrap file processing in error handling
- Add error aggregation to metrics

---

### 7. Progress Persistence
**Why:** Processing 50k files takes hours; interruptions shouldn't lose all progress.

**Implementation:**
- Save features and metrics incrementally
- Add checkpoint/resume functionality
- Track processed files to avoid reprocessing

**New files:**
- `src/data/utils/checkpoint.py`

**Features:**
```python
# Save checkpoint every N files
checkpoint_manager.save(features, metrics, processed_files)

# Resume from checkpoint
features, metrics, processed_files = checkpoint_manager.load()
```

---

### 8. Memory Optimization
**Why:** Processing very large datasets can exceed available RAM.

**Implementation:**
- Stream processing instead of loading all data
- Use memory-mapped files for large arrays
- Add batch processing with configurable batch size

**Files to modify:**
- `src/data/feature_extraction.py` - Add batch processing mode

---

### 9. Automated PDF Reporting
**Why:** Easier to share preprocessing results with collaborators.

**Implementation:**
- Generate PDF reports with matplotlib figures
- Include site comparison tables and plots
- Executive summary of preprocessing quality

**New files:**
- `src/data/preprocessing/pdf_report.py`

---

## 🟢 NICE TO HAVE - Long-term Quality

### 10. Unit Tests
**Why:** Ensure refactored code works correctly and catch regressions.

**Implementation:**
- Test each preprocessing module independently
- Test with synthetic data
- Test edge cases (single channel, very short recordings, etc.)

**New files:**
- `tests/test_preprocessing.py`
- `tests/test_feature_extraction.py`
- `tests/test_quality_metrics.py`

---

### 11. CLI Tool
**Why:** Run experiments without modifying Python scripts.

**Implementation:**
```bash
# Run filter experiment from command line
python -m src.experiments.filters \
    --data-csv datasets/ELM19/ELM19_enriched_info_filtered.csv \
    --edf-dir datasets/ELM19/raw/ELM19/ELM19_edfs \
    --config strict_both_ends \
    --workers 10 \
    --sample-size 1000
```

**New files:**
- `src/experiments/cli.py`

---

### 12. Jupyter Widgets for Interactive Parameter Tuning
**Why:** Experiment with preprocessing interactively in notebooks.

**Implementation:**
- Interactive sliders for filter parameters
- Real-time visualization of preprocessing effects
- Quick comparison of configurations

**New files:**
- `src/notebooks/interactive_preprocessing.ipynb`

---

## 🔵 RESEARCH-SPECIFIC IDEAS

### 13. ComBat Harmonization Integration
**Why:** You're already using ComBat - integrate it into the pipeline.

**Implementation:**
- Add harmonization as optional post-processing step
- Compare features before/after harmonization
- Track harmonization effectiveness

---

### 14. Site Effect Quantification Metrics
**Why:** Need objective measure of how much site effects remain.

**Implementation:**
- Compute site classification accuracy (lower = better harmonization)
- Calculate inter-site feature distribution overlap
- Track these metrics across different preprocessing strategies

**Potential metrics:**
- Site classification AUC
- Maximum Mean Discrepancy (MMD) between sites
- Wasserstein distance between site feature distributions

---

### 15. Explainable Site Effects
**Why:** Understand WHAT causes site differences, not just that they exist.

**Implementation:**
- SHAP analysis on site classification
- Identify top features that distinguish sites
- Generate interpretable reports: "Hospital X has higher gamma power in frontal channels"

---

## 📋 Implementation Priority Order

**Phase 3** (Next - High Impact):
1. Add resampling support (#1)
2. Spectral feature importance analysis (#3)
3. Better error recovery (#6)

**Phase 4** (Medium Impact):
4. Site-specific QC thresholds (#2)
5. Bad channel detection (#4)
6. Progress persistence (#7)

**Phase 5** (Long-term):
7. Temporal stability checks (#5)
8. Unit tests (#10)
9. Memory optimization (#8)

**Research Integration:**
- ComBat integration (#13)
- Site effect quantification (#14)
- Explainable site effects (#15)

---

## Notes

- Focus on improvements that directly help understand/reduce site effects
- Prioritize tools that enable experimentation (different filters, QC thresholds)
- Visualization is critical - always add plots for new metrics
- Keep backward compatibility - don't break existing workflows

---

**Last Updated:** 2025-12-12
