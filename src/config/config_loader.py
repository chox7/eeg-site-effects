"""
YAML configuration loader for ML experiments.

This module provides utilities to load YAML configuration files
and convert them into strongly-typed dataclass instances.

Note: Config files are required. Default values for optional fields
are defined in the dataclass definitions in experiment_config.py.
"""

import yaml
from pathlib import Path
from typing import TypeVar, Type, Dict, Any, get_type_hints, get_origin, get_args
from dataclasses import fields, is_dataclass

from src.config.experiment_config import (
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


def load_site_classification_config(config_path: str | Path) -> SiteClassificationConfig:
    """Load site classification configuration from YAML.

    Args:
        config_path: Path to YAML config file (required)

    Returns:
        SiteClassificationConfig instance

    Raises:
        ValueError: If config_path is None
        FileNotFoundError: If the config file doesn't exist
    """
    if config_path is None:
        raise ValueError("Config file is required. Use --config/-c to specify a YAML config file.")

    data = load_yaml(config_path)
    return dict_to_dataclass(data, SiteClassificationConfig)


def load_pathology_classification_config(config_path: str | Path) -> PathologyClassificationConfig:
    """Load pathology classification configuration from YAML.

    Args:
        config_path: Path to YAML config file (required)

    Returns:
        PathologyClassificationConfig instance

    Raises:
        ValueError: If config_path is None
        FileNotFoundError: If the config file doesn't exist
    """
    if config_path is None:
        raise ValueError("Config file is required. Use --config/-c to specify a YAML config file.")

    data = load_yaml(config_path)
    return dict_to_dataclass(data, PathologyClassificationConfig)
