"""
Configuration classes for ML experiment scripts.

This module defines configuration classes for site classification,
pathology classification, and PCA sensitivity experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PathConfig:
    """Configuration for input/output paths.

    Attributes:
        info_file: Path to the info CSV file (contains metadata)
        features_file: Path to the features CSV file
        results_file: Path to save results CSV
        pipeline_save_dir: Directory to save trained pipelines
        shap_data_save_dir: Directory to save SHAP analysis data
        log_file: Optional path for log file output
    """
    info_file: str
    features_file: str
    results_file: str
    pipeline_save_dir: Optional[str] = None
    shap_data_save_dir: Optional[str] = None
    log_file: Optional[str] = None


@dataclass
class CatBoostParams:
    """CatBoost hyperparameters.

    Attributes:
        iterations: Number of boosting iterations
        learning_rate: Learning rate
        depth: Tree depth
        l2_leaf_reg: L2 regularization coefficient
        early_stopping_rounds: Early stopping patience (optional)
        colsample_bylevel: Fraction of features for splits
        task_type: 'CPU' or 'GPU'
        thread_count: Number of threads (-1 for auto)
        random_seed: Random seed for reproducibility
        boosting_type: Boosting type ('Plain', 'Ordered')
        bootstrap_type: Bootstrap type ('Bayesian', 'Bernoulli', 'MVS', 'No')
    """
    iterations: int = 2000
    learning_rate: float = 0.1
    depth: int = 5
    l2_leaf_reg: float = 1.0
    early_stopping_rounds: Optional[int] = 50
    colsample_bylevel: Optional[float] = None
    task_type: str = "CPU"
    thread_count: int = -1
    random_seed: int = 42
    boosting_type: Optional[str] = None
    bootstrap_type: Optional[str] = None

    def to_catboost_dict(self, loss_function: str, eval_metric: Any) -> Dict[str, Any]:
        """Convert to CatBoost-compatible dictionary.

        Args:
            loss_function: Loss function name ('MultiClass', 'Logloss', etc.)
            eval_metric: Evaluation metric (string or CatBoost metric object)

        Returns:
            Dictionary of CatBoost parameters
        """
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': int(self.depth),
            'l2_leaf_reg': self.l2_leaf_reg,
            'task_type': self.task_type,
            'thread_count': self.thread_count,
            'random_seed': self.random_seed,
            'verbose': False,
            'allow_writing_files': False,
            'loss_function': loss_function,
            'eval_metric': eval_metric,
        }
        if self.early_stopping_rounds is not None:
            params['early_stopping_rounds'] = self.early_stopping_rounds
        if self.colsample_bylevel is not None:
            params['colsample_bylevel'] = self.colsample_bylevel
        if self.boosting_type is not None:
            params['boosting_type'] = self.boosting_type
        if self.bootstrap_type is not None:
            params['bootstrap_type'] = self.bootstrap_type
        return params


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration.

    Attributes:
        n_splits: Number of folds for stratified k-fold
        random_state: Random state for reproducibility
        k_calibration: Number of calibration samples (for LOSO pathology)
    """
    n_splits: int = 5
    random_state: int = 42
    k_calibration: int = 30


@dataclass
class DataConfig:
    """Data-related configuration.

    Attributes:
        covariates: List of covariate column names for harmonization (after rename)
    """
    covariates: List[str] = field(default_factory=lambda: ['age', 'gender'])


@dataclass
class SiteClassificationConfig:
    """Full configuration for site classification experiment.

    Attributes:
        paths: Input/output path configuration
        harmonization_methods: List of methods to run
        catboost_params: CatBoost hyperparameters
        cv: Cross-validation configuration
        data: Data column configuration
        experiment_name: Optional name for this experiment
    """
    paths: PathConfig
    harmonization_methods: List[str] = field(
        default_factory=lambda: ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
    )
    catboost_params: CatBoostParams = field(default_factory=CatBoostParams)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment_name: Optional[str] = None


@dataclass
class PathologyClassificationConfig:
    """Full configuration for pathology classification experiment.

    Attributes:
        paths: Input/output path configuration
        harmonization_methods: List of methods to run
        catboost_params: CatBoost hyperparameters
        cv: Cross-validation configuration
        data: Data column configuration
        ensemble_size: Number of models in GBE ensemble
        experiment_name: Optional name for this experiment
    """
    paths: PathConfig
    harmonization_methods: List[str] = field(
        default_factory=lambda: ['raw', 'sitewise', 'combat', 'neurocombat', 'covbat']
    )
    catboost_params: CatBoostParams = field(default_factory=CatBoostParams)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ensemble_size: int = 30
    experiment_name: Optional[str] = None
