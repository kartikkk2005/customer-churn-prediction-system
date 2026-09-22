"""Simple model acceptance checks used by the training pipeline."""

MIN_ROC_AUC = 0.80


def check_model_quality(roc_auc: float, minimum_roc_auc: float = MIN_ROC_AUC) -> None:
    """Raise an error when a candidate model does not meet the ROC-AUC threshold."""
    if roc_auc < minimum_roc_auc:
        raise ValueError(
            f"Model quality gate failed: ROC-AUC {roc_auc:.4f} is below "
            f"the minimum required value of {minimum_roc_auc:.2f}."
        )