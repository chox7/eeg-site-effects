# Feature Extraction with Quality Metrics - Usage Examples

## Basic Feature Extraction (existing API)

```python
from src.data.feature_extraction import map_edf_to_samples

# Extract features without metrics (fast)
features = map_edf_to_samples(
    examination_id="exam_001",
    institution_id="MOR",
    edf_dir_path="/path/to/data",
    data_group="ELM19"
)
```

## Feature Extraction with Quality Metrics (NEW!)

```python
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import generate_text_summary

# Extract features AND collect comprehensive metrics
features, metrics = map_edf_to_samples_with_metrics(
    examination_id="exam_001",
    institution_id="MOR",
    edf_dir_path="/path/to/data",
    data_group="ELM19"
)

# Print quality summary
print(generate_text_summary(metrics))

# Check if preprocessing was successful
if metrics.preprocessing_successful:
    print(f"✓ Success! Kept {metrics.segmentation.segments_kept} segments")
    print(f"  Rejection rate: {metrics.segmentation.rejection_rate:.1%}")
else:
    print(f"✗ Failed: {metrics.failure_reason}")
```

## Batch Processing with Site-Level Analysis

```python
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import (
    MetricsAggregator,
    metrics_to_dataframe,
    print_site_comparison,
    save_site_comparison_report
)

# Process multiple files and collect metrics
aggregator = MetricsAggregator()
all_features = []

for exam_id, institution_id in file_list:
    features, metrics = map_edf_to_samples_with_metrics(
        examination_id=exam_id,
        institution_id=institution_id,
        edf_dir_path=data_path,
        data_group="ELM19"
    )

    # Collect metrics
    aggregator.add_metrics(metrics)

    # Store features if successful
    if features is not None:
        all_features.append(features)

# Print site comparison
print_site_comparison(aggregator)

# Save detailed metrics to DataFrame
df = metrics_to_dataframe(aggregator.metrics_list)
df.to_csv("preprocessing_metrics.csv", index=False)

# Save site-level summary
save_site_comparison_report(aggregator, "site_comparison.csv")

# Get statistics for specific site
mor_stats = aggregator.get_site_summary("MOR")
print(f"MOR rejection rate: {mor_stats['rejection_rate_mean']:.1%}")
```

## Experiment with Different Preprocessing Configurations

```python
from src.data.config import create_custom_config
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import MetricsAggregator

# Test different quality control thresholds
configs = {
    "strict": create_custom_config(max_amplitude_uv=600.0),
    "normal": create_custom_config(max_amplitude_uv=800.0),
    "lenient": create_custom_config(max_amplitude_uv=1000.0)
}

results = {}
for config_name, config in configs.items():
    aggregator = MetricsAggregator()

    for exam_id, institution_id in sample_files:
        features, metrics = map_edf_to_samples_with_metrics(
            examination_id=exam_id,
            institution_id=institution_id,
            edf_dir_path=data_path,
            data_group="ELM19",
            preprocessing_config=config
        )
        aggregator.add_metrics(metrics)

    results[config_name] = aggregator

# Compare rejection rates across configurations
for config_name, agg in results.items():
    summary = agg.get_all_sites_summary()
    print(f"\n{config_name.upper()} Configuration:")
    for site_id, stats in summary.items():
        print(f"  {site_id}: {stats['rejection_rate_mean']:.1%} rejection")
```

## Site-Specific Configuration

```python
from src.data.config import get_site_specific_config
from src.data.feature_extraction import map_edf_to_samples_with_metrics

# Use site-specific config (e.g., for 60 Hz line noise in US)
config = get_site_specific_config("US_SITE")

features, metrics = map_edf_to_samples_with_metrics(
    examination_id="exam_001",
    institution_id="US_SITE",
    edf_dir_path="/path/to/data",
    data_group="ELM19",
    preprocessing_config=config
)
```

## Parallel Processing with Metrics

```python
from multiprocessing import Pool
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import MetricsAggregator

def process_file(args):
    exam_id, institution_id, edf_dir_path, data_group = args
    return map_edf_to_samples_with_metrics(
        exam_id, institution_id, edf_dir_path, data_group
    )

# Prepare arguments
args_list = [
    (exam_id, inst_id, data_path, "ELM19")
    for exam_id, inst_id in file_list
]

# Process in parallel
with Pool(processes=8) as pool:
    results = pool.map(process_file, args_list)

# Aggregate metrics
aggregator = MetricsAggregator()
all_features = []

for features, metrics in results:
    aggregator.add_metrics(metrics)
    if features is not None:
        all_features.append(features)

print_site_comparison(aggregator)
```

## Access Metrics Programmatically

```python
features, metrics = map_edf_to_samples_with_metrics(...)

# Access specific metrics
print(f"Sampling frequency: {metrics.sampling_frequency_hz} Hz")
print(f"Processing time: {metrics.processing_duration_seconds:.2f}s")
print(f"Raw signal mean amplitude: {metrics.raw_signal_quality.mean_amplitude_uv:.2f} µV")
print(f"Preprocessed signal MAD: {metrics.preprocessed_signal_quality.median_mad_uv:.2f} µV")
print(f"Segments kept: {metrics.segmentation.segments_kept}/{metrics.segmentation.total_segments}")

# Check warnings
if metrics.warnings:
    print("Warnings:")
    for warning in metrics.warnings:
        print(f"  - {warning}")

# Export to JSON
with open("metrics.json", "w") as f:
    f.write(metrics.to_json())

# Export to dict
metrics_dict = metrics.to_dict()
```

## What Metrics Are Collected?

### Signal Quality Metrics
- Mean, std, max, min amplitude (µV)
- Median MAD (Median Absolute Deviation)
- Collected for both raw and preprocessed signals

### Segmentation Metrics
- Total segments created
- Segments rejected (amplitude threshold)
- Segments rejected (flatness threshold)
- Total segments kept
- Rejection rate

### Processing Information
- Processing timestamp
- Processing duration
- Sampling frequency
- Recording duration
- Number of channels
- Filters applied

### Quality Flags
- Success/failure status
- Warnings (e.g., high rejection rate)
- Failure reasons (if failed)

This information is crucial for identifying site-specific preprocessing issues!

---

# Preprocessing Visualization - Usage Examples

## Visualize Single File Preprocessing

```python
import mne
from src.data.preprocessing import (
    preprocess_and_segment_with_metrics,
    plot_signal_comparison,
    plot_power_spectrum_comparison,
    plot_preprocessing_summary,
    save_preprocessing_report
)
from src.data.config import DEFAULT_PREPROCESSING_CONFIG

# Load EDF file
raw_edf = mne.io.read_raw_edf("path/to/file.edf", preload=True)

# Preprocess with metrics
clean_segments, time_indices, preprocessed_edf, metrics = preprocess_and_segment_with_metrics(
    raw_edf,
    DEFAULT_PREPROCESSING_CONFIG,
    examination_id="exam_001",
    institution_id="MOR",
    data_group="ELM19"
)

# 1. Compare raw vs preprocessed signals
fig = plot_signal_comparison(
    raw_edf,
    preprocessed_edf,
    channels=['Cz', 'Pz', 'Fz'],  # Select channels to visualize
    duration=10.0,  # 10 seconds
    start_time=30.0  # Starting at 30s
)
fig.savefig('signal_comparison.png', dpi=150)

# 2. Compare power spectra
fig = plot_power_spectrum_comparison(
    raw_edf,
    preprocessed_edf,
    channel='Cz',
    fmax=50.0  # Show up to 50 Hz
)
fig.savefig('spectrum_comparison.png', dpi=150)

# 3. Generate comprehensive preprocessing summary
fig = plot_preprocessing_summary(
    raw_edf,
    preprocessed_edf,
    metrics,
    channel='Cz'
)
fig.savefig('preprocessing_summary.png', dpi=150)

# Or use the convenience function:
save_preprocessing_report(
    raw_edf,
    preprocessed_edf,
    metrics,
    'preprocessing_report.png',
    channel='Cz'
)
```

## Site Comparison Visualizations

```python
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import (
    MetricsAggregator,
    plot_rejection_rates_by_site,
    plot_signal_quality_by_site,
    plot_site_comparison_dashboard,
    plot_metric_distribution_by_site
)

# Process multiple files across sites
aggregator = MetricsAggregator()

for exam_id, institution_id in file_list:
    features, metrics = map_edf_to_samples_with_metrics(
        examination_id=exam_id,
        institution_id=institution_id,
        edf_dir_path=data_path,
        data_group="ELM19"
    )
    aggregator.add_metrics(metrics)

# 1. Compare rejection rates across sites
fig = plot_rejection_rates_by_site(aggregator)
fig.savefig('rejection_rates_by_site.png', dpi=150)

# 2. Compare signal quality across sites
fig = plot_signal_quality_by_site(aggregator)
fig.savefig('signal_quality_by_site.png', dpi=150)

# 3. Comprehensive dashboard with all comparisons
fig = plot_site_comparison_dashboard(aggregator)
fig.savefig('site_comparison_dashboard.png', dpi=150)

# 4. Box plot showing distribution of rejection rates
fig = plot_metric_distribution_by_site(
    aggregator.metrics_list,
    metric_name='rejection_rate'
)
fig.savefig('rejection_distribution.png', dpi=150)

# 5. Box plot showing distribution of signal amplitudes
fig = plot_metric_distribution_by_site(
    aggregator.metrics_list,
    metric_name='mean_amplitude'
)
fig.savefig('amplitude_distribution.png', dpi=150)
```

## Compare Different Preprocessing Strategies Visually

```python
from src.data.config import create_custom_config
from src.data.preprocessing import (
    MetricsAggregator,
    plot_rejection_rates_by_site
)
import matplotlib.pyplot as plt

# Test different configurations
configs = {
    "strict_qc": create_custom_config(max_amplitude_uv=600.0),
    "normal_qc": create_custom_config(max_amplitude_uv=800.0),
    "lenient_qc": create_custom_config(max_amplitude_uv=1000.0)
}

results = {}
for config_name, config in configs.items():
    aggregator = MetricsAggregator()

    for exam_id, institution_id in sample_files:
        features, metrics = map_edf_to_samples_with_metrics(
            examination_id=exam_id,
            institution_id=institution_id,
            edf_dir_path=data_path,
            data_group="ELM19",
            preprocessing_config=config
        )
        aggregator.add_metrics(metrics)

    results[config_name] = aggregator

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (config_name, agg) in enumerate(results.items()):
    summary = agg.get_all_sites_summary()
    sites = list(summary.keys())
    rejection_rates = [summary[s]['rejection_rate_mean'] * 100 for s in sites]

    axes[i].bar(range(len(sites)), rejection_rates, alpha=0.7)
    axes[i].set_title(f'{config_name}')
    axes[i].set_ylabel('Rejection Rate (%)')
    axes[i].set_xticks(range(len(sites)))
    axes[i].set_xticklabels(sites, rotation=45, ha='right')
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('config_comparison.png', dpi=150)
```

## Interactive Exploration in Jupyter Notebook

```python
# In a Jupyter notebook
%matplotlib inline

from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import (
    plot_preprocessing_summary,
    generate_text_summary
)

# Process a file
features, metrics = map_edf_to_samples_with_metrics(
    examination_id="exam_001",
    institution_id="MOR",
    edf_dir_path="/path/to/data",
    data_group="ELM19"
)

# Print text summary
print(generate_text_summary(metrics))

# Show interactive plots
plot_preprocessing_summary(raw_edf, preprocessed_edf, metrics)
plt.show()

# Quick quality check
print(metrics.quality_summary)
```

## Batch Report Generation

```python
import os
from pathlib import Path
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import save_preprocessing_report
import mne

# Create output directory
output_dir = Path("preprocessing_reports")
output_dir.mkdir(exist_ok=True)

# Process files and generate reports
for exam_id, institution_id in file_list:
    print(f"Processing {exam_id}...")

    # Load file
    if data_group == "ELM19":
        edf_path = f"{edf_dir_path}/{exam_id}.edf"
    else:
        edf_path = get_edf_path(exam_id)

    raw_edf = mne.io.read_raw_edf(edf_path, preload=True)

    # Process with metrics
    features, metrics = map_edf_to_samples_with_metrics(
        examination_id=exam_id,
        institution_id=institution_id,
        edf_dir_path=edf_dir_path,
        data_group=data_group
    )

    # Save individual report
    if metrics.preprocessing_successful:
        report_path = output_dir / f"{institution_id}_{exam_id}_report.png"
        save_preprocessing_report(
            raw_edf,
            preprocessed_edf,
            metrics,
            str(report_path)
        )

print(f"Reports saved to {output_dir}")
```

These visualization tools help you quickly identify site-specific preprocessing issues!
