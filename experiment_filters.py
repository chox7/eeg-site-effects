"""
Experiment with different filter configurations to reduce site effects.

Saves features for each config to datasets/ELM19/experiments/ for ML comparison.
"""

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import json
from pathlib import Path

from src.data.config import FilterConfig, PreprocessingConfig
from src.data import map_edf_to_samples_with_metrics
from src.data.preprocessing import (
    MetricsAggregator,
    print_site_comparison,
    plot_site_comparison_dashboard,
    metrics_to_dataframe
)
from src.utils.utils import get_feat_names

logging.basicConfig(level=logging.WARNING)


# =============================================================================
# DEFINE FILTER CONFIGURATIONS TO TEST
# =============================================================================

FILTER_CONFIGS = {
    "original": {
        "description": "Original preprocessing from old code",
        "filters": [
            FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
            FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
            FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
        ]
    },

    "remove_low_freq_artifacts": {
        "description": "Stricter highpass to remove drift and slow artifacts",
        "filters": [
            FilterConfig(type='highpass', f_pass=1.0, f_stop=0.3),
            FilterConfig(type='lowpass', f_pass=40.0, f_stop=50.0),
            FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
        ]
    },

    "remove_high_freq_artifacts": {
        "description": "Stricter lowpass to remove muscle and EMG artifacts",
        "filters": [
            FilterConfig(type='highpass', f_pass=0.5, f_stop=0.1),
            FilterConfig(type='lowpass', f_pass=30.0, f_stop=35.0),
            FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
        ]
    },

    "strict_both_ends": {
        "description": "Aggressive filtering on both ends: 1-30 Hz",
        "filters": [
            FilterConfig(type='highpass', f_pass=1.0, f_stop=0.3),
            FilterConfig(type='lowpass', f_pass=30.0, f_stop=35.0),
            FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
        ]
    },

    "clinical_range": {
        "description": "Very clean clinical EEG range: 1-25 Hz",
        "filters": [
            FilterConfig(type='highpass', f_pass=1.0, f_stop=0.5),
            FilterConfig(type='lowpass', f_pass=25.0, f_stop=30.0),
            FilterConfig(type='notch', notch_freq=50.0, notch_widths=10.0)
        ]
    },
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
        print(e)
        return idx, None, None


# =============================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================

def run_filter_experiment(
    df_info,
    edf_dir="datasets/ELM19/raw/ELM19/ELM19_edfs",
    data_group="ELM19",
    output_base="datasets/ELM19/experiments",
    num_workers=10,
    sample_size=None
):
    """
    Run filter experiment and save features for ML.

    Args:
        df_info: DataFrame with examination info
        edf_dir: Path to EDF files
        data_group: Dataset name
        output_base: Base directory for experiment outputs
        num_workers: Number of parallel workers
        sample_size: Number of files to test (None = all)
    """

    if sample_size is not None:
        print(f"Testing on {sample_size} files (sample)")
        df_info = df_info.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"Processing all {len(df_info)} files")

    base_path = Path(output_base)
    base_path.mkdir(parents=True, exist_ok=True)

    results_by_config = {}

    for config_name, config_info in FILTER_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {config_name}")
        print(f"Description: {config_info['description']}")
        print(f"{'='*80}")

        # Create output directory for this config
        config_dir = base_path / config_name
        config_dir.mkdir(exist_ok=True)

        # Create preprocessing config
        preproc_config = PreprocessingConfig(
            filters=config_info['filters'],
            desired_sampling_freq=100.0
        )

        # Process files in parallel
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_file_with_config)(
                df_info.examination_id.iloc[idx],
                df_info.institution_id.iloc[idx],
                idx,
                edf_dir,
                data_group,
                preproc_config
            )
            for idx in tqdm(df_info.index, desc=f"{config_name}")
        )

        # Sort by index
        results.sort(key=lambda x: x[0])

        # Aggregate
        aggregator = MetricsAggregator()
        features_list = []

        for idx, features, metrics in results:
            if metrics is not None:
                aggregator.add_metrics(metrics)
            if features is not None:
                features_list.append(features)

        print(f"✓ Success: {len(features_list)}/{len(df_info)} files")

        # SAVE FEATURES FOR ML
        if len(features_list) > 0:
            df_feats = pd.DataFrame(features_list, columns=get_feat_names())
            features_path = config_dir / "features.csv"
            df_feats.to_csv(features_path, index=False)
            print(f"  → Features saved: {features_path}")

        # Save metrics
        metrics_df = metrics_to_dataframe(aggregator.metrics_list)
        metrics_path = config_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  → Metrics saved: {metrics_path}")

        # Save config for reproducibility
        config_dict = {
            'description': config_info['description'],
            'filters': [
                {
                    'type': f.type,
                    'f_pass': f.f_pass,
                    'f_stop': f.f_stop,
                    'notch_freq': f.notch_freq,
                    'notch_widths': f.notch_widths
                }
                for f in config_info['filters']
            ],
            'desired_sampling_freq': 100.0
        }
        config_path = config_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        results_by_config[config_name] = {
            'aggregator': aggregator,
            'features': features_list,
            'n_successful': len(features_list),
            'n_total': len(df_info),
            'output_dir': config_dir
        }

    return results_by_config, base_path


# =============================================================================
# COMPARISON
# =============================================================================

def create_comparison_summary(results_by_config, base_path):
    """Create comparison summary across all configs."""

    print("\n" + "="*80)
    print("CREATING COMPARISON SUMMARY")
    print("="*80)

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

    # Save comparison
    comparison_path = base_path / "comparison_summary.csv"
    df_comparison.to_csv(comparison_path, index=False)
    print(f"✓ Saved: {comparison_path}")

    # Print summary
    for config_name in results_by_config.keys():
        print(f"\n{config_name.upper()}:")
        print_site_comparison(results_by_config[config_name]['aggregator'])

    return df_comparison


def create_visualizations(results_by_config, base_path):
    """Create comparison visualizations."""
    import matplotlib.pyplot as plt

    print(f"\nCreating visualizations...")

    for config_name, result in results_by_config.items():
        if len(result['aggregator'].metrics_list) > 0:
            fig = plot_site_comparison_dashboard(result['aggregator'])
            fig.suptitle(f'{config_name}', fontsize=16, fontweight='bold')

            plot_path = result['output_dir'] / "site_comparison_dashboard.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"✓ Visualizations saved in each experiment folder")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("="*80)
    print("FILTER EXPERIMENT - Extract Features for ML Comparison")
    print("="*80)

    # Load dataset
    df_info = pd.read_csv('data/ELM19/filtered/ELM19_info_filtered.csv')
    df_info = df_info.reset_index(drop=True)

    print(f"\nDataset: {len(df_info)} files")
    print(f"Sites: {df_info['institution_id'].unique()}")

    # Run experiment
    # IMPORTANT: Set sample_size=None to process ALL files for real experiments
    # Use sample_size=100 for quick testing
    results, base_path = run_filter_experiment(
        df_info,
        edf_dir="data/ELM19/raw/ELM19/ELM19_edfs",
        data_group="ELM19",
        output_base="data/ELM19/experiments",
        num_workers=24,
        sample_size=None  # Change to None for full dataset!
    )

    # Create comparison
    comparison_df = create_comparison_summary(results, base_path)

    # Create visualizations
    create_visualizations(results, base_path)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {base_path}/")
    print("\nFor each filter config you have:")
    print("  - features.csv       # Use this for training ML models")
    print("  - metrics.csv        # Preprocessing quality metrics")
    print("  - config.json        # Exact config for reproducibility")
    print("\nNext steps:")
    print("1. Train ML models on each features.csv")
    print("2. Compare model performance across configs")
    print("3. Choose config that gives best site effect reduction")
