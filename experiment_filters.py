"""
Experiment with different filter configurations to reduce site effects.

Based on your existing workflow in 03_preprocessing_and_feature_extraction.ipynb
"""

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging

from src.data.config import FilterConfig, PreprocessingConfig
from src.data.feature_extraction import map_edf_to_samples_with_metrics
from src.data.preprocessing import (
    MetricsAggregator,
    print_site_comparison,
    plot_site_comparison_dashboard,
    metrics_to_dataframe
)

logging.basicConfig(level=logging.WARNING)


# =============================================================================
# DEFINE FILTER CONFIGURATIONS TO TEST
# =============================================================================

FILTER_CONFIGS = {
    "original": [
        FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
        FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ],

    "remove_low_freq_artifacts": [
        # Remove very low frequency artifacts (drift, slow movement)
        FilterConfig(type='highpass', f_pass=1.0, f_stop=0.3),
        FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ],

    "remove_high_freq_artifacts": [
        # Remove high frequency artifacts (muscle, EMG)
        FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
        FilterConfig(type='lowpass', f_pass=30.0, f_stop=35.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ],

    "strict_both_ends": [
        # Aggressive filtering: 1-30 Hz
        FilterConfig(type='highpass', f_pass=1.0, f_stop=0.3),
        FilterConfig(type='lowpass', f_pass=30.0, f_stop=35.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ],

    "clinical_range": [
        # Very clean clinical EEG range: 1-25 Hz
        FilterConfig(type='highpass', f_pass=1.0, f_stop=0.5),
        FilterConfig(type='lowpass', f_pass=25.0, f_stop=30.0),
        FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
    ],
}


# =============================================================================
# HELPER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================

def process_single_file_with_config(exam_id, institution_id, idx, edf_dir, data_group, config):
    """Process a single file with given configuration."""
    try:
        features, metrics = map_edf_to_samples_with_metrics(
            examination_id=exam_id,
            institution_id=institution_id,
            edf_dir_path=edf_dir,
            data_group=data_group,
            preprocessing_config=config
        )
        return idx, features, metrics
    except Exception as e:
        print(f"Error processing {exam_id}: {e}")
        return idx, None, None


# =============================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================

def run_filter_experiment(
    df_info,
    edf_dir="datasets/ELM19/raw/ELM19/ELM19_edfs",
    data_group="ELM19",
    num_workers=10,
    sample_size=None  # Set to small number for testing, None for all
):
    """
    Run the filter experiment on your dataset.

    Args:
        df_info: DataFrame with examination info (from CSV)
        edf_dir: Path to EDF files
        data_group: Dataset name ("ELM19" or "TUH")
        num_workers: Number of parallel workers
        sample_size: Number of files to test (None = all files)
    """

    # Sample data if requested
    if sample_size is not None:
        print(f"Testing on {sample_size} files (sample)")
        df_info = df_info.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"Testing on all {len(df_info)} files")

    results_by_config = {}

    for config_name, filters in FILTER_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Testing configuration: {config_name}")
        print(f"{'='*80}")

        # Create preprocessing config
        preproc_config = PreprocessingConfig(
            filters=filters,
            desired_sampling_freq=100.0
        )

        # Process files in parallel (like your original notebook)
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_file_with_config)(
                df_info.examination_id.iloc[idx],
                df_info.institution_id.iloc[idx],
                idx,
                edf_dir,
                data_group,
                preproc_config
            )
            for idx in tqdm(df_info.index, desc=f"Processing {config_name}")
        )

        # Sort by index
        results.sort(key=lambda x: x[0])

        # Aggregate metrics
        aggregator = MetricsAggregator()
        features_list = []

        for idx, features, metrics in results:
            if metrics is not None:
                aggregator.add_metrics(metrics)
            if features is not None:
                features_list.append(features)

        results_by_config[config_name] = {
            'aggregator': aggregator,
            'features': features_list,
            'n_successful': len(features_list),
            'n_total': len(df_info)
        }

        print(f"✓ Processed: {len(features_list)}/{len(df_info)} files successful")

    return results_by_config


# =============================================================================
# COMPARISON AND ANALYSIS
# =============================================================================

def compare_configurations(results_by_config):
    """Compare all filter configurations."""

    print("\n" + "="*80)
    print("COMPARISON ACROSS FILTER CONFIGURATIONS")
    print("="*80)

    # Print site comparison for each config
    for config_name, result in results_by_config.items():
        print(f"\n{config_name.upper()}:")
        print(f"Success rate: {result['n_successful']}/{result['n_total']}")
        print("-" * 80)
        print_site_comparison(result['aggregator'])

    # Create comparison DataFrame
    comparison_data = []
    for config_name, result in results_by_config.items():
        aggregator = result['aggregator']
        site_summary = aggregator.get_all_sites_summary()

        for site_id, stats in site_summary.items():
            comparison_data.append({
                'filter_config': config_name,
                'site': site_id,
                'n_files': stats['total_files'],
                'success_rate': stats['successful'] / stats['total_files'] if stats['total_files'] > 0 else 0,
                'rejection_rate_mean': stats['rejection_rate_mean'],
                'rejection_rate_std': stats['rejection_rate_std'],
                'signal_amp_mean': stats['mean_amplitude_mean'],
                'signal_amp_std': stats['mean_amplitude_std'],
            })

    df_comparison = pd.DataFrame(comparison_data)

    # Save results
    df_comparison.to_csv("filter_experiment_comparison.csv", index=False)
    print(f"\n✓ Comparison saved to: filter_experiment_comparison.csv")

    # Also save detailed metrics for each config
    for config_name, result in results_by_config.items():
        metrics_df = metrics_to_dataframe(result['aggregator'].metrics_list)
        metrics_df.to_csv(f"filter_experiment_{config_name}_detailed.csv", index=False)

    return df_comparison


def create_visualizations(results_by_config):
    """Create comparison visualizations."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path("filter_experiment_plots")
    output_dir.mkdir(exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}/...")

    # Dashboard for each configuration
    for config_name, result in results_by_config.items():
        if len(result['aggregator'].metrics_list) > 0:
            fig = plot_site_comparison_dashboard(result['aggregator'])
            fig.suptitle(f'Filter Config: {config_name}', fontsize=16, fontweight='bold')
            filepath = output_dir / f"dashboard_{config_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"✓ Visualizations saved to {output_dir}/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Load your dataset info
    print("Loading dataset info...")
    df_info = pd.read_csv('datasets/ELM19/ELM19_enriched_info_filtered.csv')
    df_info = df_info.reset_index(drop=True)

    print(f"Loaded {len(df_info)} files")
    print(f"Institutions: {df_info['institution_id'].unique()}")

    # Run experiment
    # NOTE: Set sample_size to a small number (e.g., 100) for testing!
    # Set to None to process all files
    results = run_filter_experiment(
        df_info,
        edf_dir="datasets/ELM19/raw/ELM19/ELM19_edfs",
        data_group="ELM19",
        num_workers=10,
        sample_size=100  # Start with 100 files for testing
    )

    # Compare results
    comparison_df = compare_configurations(results)

    # Create visualizations
    create_visualizations(results)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nReview:")
    print("1. filter_experiment_comparison.csv - Summary comparison")
    print("2. filter_experiment_plots/ - Visual comparisons")
    print("3. Look for which config reduces site differences most")
