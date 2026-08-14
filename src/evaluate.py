"""
Model Evaluation Module for Customer Churn Prediction.
Calculates key performance metrics, confusion matrices, and ROC curves.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for classification models."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(float(val), 4) for val in fpr],
            "tpr": [round(float(val), 4) for val in tpr],
        },
    }
