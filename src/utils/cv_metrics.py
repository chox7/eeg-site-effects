"""
Cross-validation metrics and utilities for classification tasks.

This module provides functions for computing classification metrics and
creating cross-validation splits for multi-center EEG data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def get_scores_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Compute classification metrics for multi-class problems.

    Parameters
    ----------
    y_true : array-like
        True labels (original class names, not encoded).
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    label_encoder : LabelEncoder
        Fitted label encoder for class names.

    Returns
    -------
    dict
        Dictionary containing:
        - MCC_Overall: Overall Matthews Correlation Coefficient
        - Accuracy: Classification accuracy
        - Precision (Macro/Weighted): Precision scores
        - Recall (Macro/Weighted): Recall scores
        - F1-Score (Macro/Weighted): F1 scores
        - AUC (OvR/OvO): Area under ROC curve
        - MCC_{class}: Per-class MCC scores
    """
    y_pred = label_encoder.inverse_transform(np.argmax(y_prob, axis=1))

    scores = {
        "MCC_Overall": matthews_corrcoef(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Precision (Weighted)": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall (Macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (Weighted)": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1-Score (Macro)": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1-Score (Weighted)": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "AUC (OvR)": roc_auc_score(y_true, y_prob, multi_class="ovr"),
        "AUC (OvO)": roc_auc_score(y_true, y_prob, multi_class="ovo"),
    }

    # Per-class MCC
    mcc_per_class = {
        f"MCC_{cls}": matthews_corrcoef(
            (y_true == cls).astype(int), (y_pred == cls).astype(int)
        )
        for cls in label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    }
    scores.update(mcc_per_class)

    return scores


def get_scores_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics for binary problems.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float, default=0.5
        Classification threshold.

    Returns
    -------
    dict
        Dictionary containing MCC, Accuracy, Precision, Recall, F1, and AUC.
    """
    y_pred = (y_prob > threshold).astype(int)

    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    """
    Compute confusion matrix from predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True labels (original class names, not encoded).
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    label_encoder : LabelEncoder
        Fitted label encoder for class names.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    """
    y_pred = label_encoder.inverse_transform(np.argmax(y_prob, axis=1))
    labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def stratified_site_folds(
    df_info: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    site_column: str = "institution_id",
) -> list:
    """
    Create cross-validation folds where each fold contains data from all sites.

    This ensures that every site is represented in both training and test sets
    for each fold, with random splits within each site.

    Parameters
    ----------
    df_info : pd.DataFrame
        DataFrame containing site information.
    n_splits : int, default=5
        Number of folds.
    seed : int, default=42
        Random seed for reproducibility.
    site_column : str, default="institution_id"
        Column name containing site identifiers.

    Returns
    -------
    list of tuples
        List of (train_indices, test_indices) for each fold.
    """
    np.random.seed(seed)
    site_ids = df_info[site_column].unique()

    # Get indices for each site
    indices_per_site = {
        site: df_info[df_info[site_column] == site].index.values
        for site in site_ids
    }

    # Randomly split each site's data into folds
    site_folds = {
        site: np.array_split(np.random.permutation(idxs), n_splits)
        for site, idxs in indices_per_site.items()
    }

    # Combine site folds into full train/test splits
    folds = []
    for fold_idx in range(n_splits):
        test_indices = []
        for site in site_ids:
            test_indices.extend(site_folds[site][fold_idx])
        test_indices = np.array(test_indices)
        train_indices = df_info.index.difference(test_indices)
        folds.append((train_indices.to_numpy(), test_indices))

    return folds
