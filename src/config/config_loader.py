"""
YAML configuration loader for ML experiments.

This module provides utilities to load YAML configuration files
and convert them into strongly-typed dataclass instances.
"""

import yaml
from pathlib import Path
from typing import TypeVar, Type, Dict, Any, Optional, get_type_hints, get_origin, get_args
from dataclasses import fields, is_dataclass

from src.config.experiment_config import (
    PathConfig,
    CatBoostParams,
    CrossValidationConfig,
    DataConfig,
    SiteClassificationConfig,
    PathologyClassificationConfig,
)


T = TypeVar('T')


def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _get_dataclass_type(field_type):
    """Extract the actual dataclass type from a type hint."""
    # Handle Optional[SomeDataclass]
    origin = get_origin(field_type)
    if origin is type(None):
        return None

    # Check if it's a Union (Optional is Union[X, None])
    if origin:
        args = get_args(field_type)
        for arg in args:
            if is_dataclass(arg):
                return arg
        return None

    # Direct dataclass type
    if is_dataclass(field_type):
        return field_type

    return None


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance, handling nested dataclasses.

    Args:
        data: Dictionary containing configuration values
        cls: The dataclass type to instantiate

    Returns:
        An instance of the specified dataclass
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    # Get field information
    field_info = {f.name: f for f in fields(cls)}

    # Try to get type hints for better type resolution
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(cls)}

    processed_data = {}

    for key, value in data.items():
        if key not in field_info:
            continue

        field_type = type_hints.get(key, field_info[key].type)

        # Handle nested dataclasses
        dataclass_type = _get_dataclass_type(field_type)
        if dataclass_type and isinstance(value, dict):
            processed_data[key] = dict_to_dataclass(value, dataclass_type)
        else:
            processed_data[key] = value

    return cls(**processed_data)


# Default configurations (matching current hardcoded values)

DEFAULT_SITE_CLASSIFICATION_PATHS = PathConfig(
    info_file='data/ELM19/filtered/ELM19_info_filtered_norm.csv',
    features_file='data/ELM19/filtered/ELM19_features_filtered_norm.csv',
    results_file='results/tables/05_experiment_filters/exp01_site_clf_results.csv',
    pipeline_save_dir='models/05_experiment_filters/exp01_site_clf_pipelines',
    shap_data_save_dir='results/shap_data/05_experiment_filters/exp01_site_clf',
)

DEFAULT_SITE_CATBOOST_PARAMS = CatBoostParams(
    iterations=2000,
    learning_rate=0.2136106733298358,
    depth=5,
    l2_leaf_reg=1.0050061307458207,
    early_stopping_rounds=50,
    task_type="GPU",
    thread_count=20,
)

DEFAULT_PATHOLOGY_CLASSIFICATION_PATHS = PathConfig(
    info_file='data/ELM19/filtered/ELM19_info_filtered.csv',
    features_file='data/ELM19/filtered/ELM19_features_filtered.csv',
    results_file='results/tables/05_experiment_filters/exp02_patho_clf_results.csv',
    pipeline_save_dir='models/05_experiment_filters/exp02_patho_clf_pipelines',
    shap_data_save_dir='results/shap_data/05_experiment_filters/exp02_patho_clf',
)

DEFAULT_PATHOLOGY_CATBOOST_PARAMS = CatBoostParams(
    iterations=700,
    learning_rate=0.08519504279364008,
    depth=6,
    l2_leaf_reg=1.1029971156522604,
    colsample_bylevel=0.019946626267165004,
    task_type="CPU",
    thread_count=-1,
    boosting_type='Plain',
    bootstrap_type='MVS',
)


def load_site_classification_config(
    config_path: Optional[str | Path] = None
) -> SiteClassificationConfig:
    """Load site classification configuration from YAML.

    Args:
        config_path: Path to YAML config file. If None, returns default config.

    Returns:
        SiteClassificationConfig instance
    """
    if config_path is None:
        return SiteClassificationConfig(
            paths=DEFAULT_SITE_CLASSIFICATION_PATHS,
            catboost_params=DEFAULT_SITE_CATBOOST_PARAMS,
        )

    data = load_yaml(config_path)
    return dict_to_dataclass(data, SiteClassificationConfig)


def load_pathology_classification_config(
    config_path: Optional[str | Path] = None
) -> PathologyClassificationConfig:
    """Load pathology classification configuration from YAML.

    Args:
        config_path: Path to YAML config file. If None, returns default config.

    Returns:
        PathologyClassificationConfig instance
    """
    if config_path is None:
        return PathologyClassificationConfig(
            paths=DEFAULT_PATHOLOGY_CLASSIFICATION_PATHS,
            catboost_params=DEFAULT_PATHOLOGY_CATBOOST_PARAMS,
        )

    data = load_yaml(config_path)
    return dict_to_dataclass(data, PathologyClassificationConfig)
