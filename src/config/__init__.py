"""Configuration module for ML experiments."""

from src.config.experiment_config import (
    PathConfig,
    CatBoostParams,
    CrossValidationConfig,
    DataConfig,
    SiteClassificationConfig,
    PathologyClassificationConfig,
)

from src.config.config_loader import (
    load_yaml,
    dict_to_dataclass,
    load_site_classification_config,
    load_pathology_classification_config,
)

__all__ = [
    # Dataclasses
    'PathConfig',
    'CatBoostParams',
    'CrossValidationConfig',
    'DataConfig',
    'SiteClassificationConfig',
    'PathologyClassificationConfig',
    # Loaders
    'load_yaml',
    'dict_to_dataclass',
    'load_site_classification_config',
    'load_pathology_classification_config',
]
