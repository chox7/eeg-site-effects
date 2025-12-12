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
