import numpy as np
import pandas as pd
import shap


def compute_shap_from_pipeline(pipeline, X_test):
    """
    Compute SHAP values from a fitted sklearn Pipeline.

    Handles two pipeline structures:
    - With 'harmonize' step: transforms X through the harmonizer first.
    - Without 'harmonize' step (raw): uses X_test directly.

    Works with both GBE ensemble models (which expose ``compute_shap_values``)
    and single CatBoost/tree models (uses ``shap.TreeExplainer``).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline. Must have a ``clf`` step.
    X_test : pd.DataFrame
        Test features before any harmonization.

    Returns
    -------
    shap_explanation : shap.Explanation
        Shape ``(n_samples, n_features)`` for binary/GBE, or
        ``(n_samples, n_features, n_classes)`` for multiclass.
    X_transformed : pd.DataFrame
        The (harmonized) features used for the SHAP computation.
    """
    model = pipeline.named_steps['clf']

    if 'harmonize' in pipeline.named_steps:
        X_arr = pipeline.named_steps['harmonize'].transform(X_test)
        X_transformed = pd.DataFrame(X_arr, columns=model.feature_names_)
    else:
        X_transformed = (
            X_test[model.feature_names_]
            if isinstance(X_test, pd.DataFrame)
            else pd.DataFrame(X_test, columns=model.feature_names_)
        )

    if hasattr(model, 'compute_shap_values'):
        shap_explanation = model.compute_shap_values(X_transformed)
    else:
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_transformed)

    return shap_explanation, X_transformed


def shap_to_mean_series(shap_explanation):
    """
    Reduce a SHAP Explanation to mean absolute SHAP per feature.

    Handles both binary ``(n_samples, n_features)`` and
    multiclass ``(n_samples, n_features, n_classes)`` shapes.

    Returns
    -------
    pd.Series
        Mean absolute SHAP value indexed by feature name.
    """
    values = shap_explanation.values
    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=shap_explanation.feature_names)


def shap_to_per_class_series(shap_explanation, classes):
    """
    For multiclass SHAP, return mean absolute SHAP per feature for each class.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        Shape ``(n_samples, n_features, n_classes)``.
    classes : array-like
        Class labels corresponding to the last axis.

    Returns
    -------
    dict[Any, pd.Series]
        Mapping from class label to mean absolute SHAP Series.
    """
    values = shap_explanation.values        # (n_samples, n_features, n_classes)
    mean_abs = np.mean(np.abs(values), axis=0)  # (n_features, n_classes)
    return {
        class_label: pd.Series(mean_abs[:, idx], index=shap_explanation.feature_names)
        for idx, class_label in enumerate(classes)
    }