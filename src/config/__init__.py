"""Configuration module for ML experiments."""

import yaml
from pathlib import Path
from typing import TypeVar, Type, Dict, Any, get_type_hints, get_origin, get_args
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
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _get_dataclass_type(field_type):
    origin = get_origin(field_type)
    if origin is type(None):
        return None
    if origin:
        args = get_args(field_type)
        for arg in args:
            if is_dataclass(arg):
                return arg
        return None
    if is_dataclass(field_type):
        return field_type
    return None


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    field_info = {f.name: f for f in fields(cls)}
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(cls)}
    processed_data = {}
    for key, value in data.items():
        if key not in field_info:
            continue
        field_type = type_hints.get(key, field_info[key].type)
        dataclass_type = _get_dataclass_type(field_type)
        if dataclass_type and isinstance(value, dict):
            processed_data[key] = dict_to_dataclass(value, dataclass_type)
        else:
            processed_data[key] = value
    return cls(**processed_data)


def load_site_classification_config(config_path: str | Path) -> SiteClassificationConfig:
    if config_path is None:
        raise ValueError("Config file is required. Use --config/-c to specify a YAML config file.")
    data = load_yaml(config_path)
    return dict_to_dataclass(data, SiteClassificationConfig)


def load_pathology_classification_config(config_path: str | Path) -> PathologyClassificationConfig:
    if config_path is None:
        raise ValueError("Config file is required. Use --config/-c to specify a YAML config file.")
    data = load_yaml(config_path)
    return dict_to_dataclass(data, PathologyClassificationConfig)


__all__ = [
    'PathConfig',
    'CatBoostParams',
    'CrossValidationConfig',
    'DataConfig',
    'SiteClassificationConfig',
    'PathologyClassificationConfig',
    'load_yaml',
    'dict_to_dataclass',
    'load_site_classification_config',
    'load_pathology_classification_config',
]
