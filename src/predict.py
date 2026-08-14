"""
Inference Module for Single & Batch Customer Churn Prediction.
"""

import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import add_engineered_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

DEFAULT_MODEL_PATH = os.path.join("models", "best_model.joblib")
DEFAULT_PREPROCESSOR_PATH = os.path.join("models", "preprocessor.joblib")


def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH, preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH
) -> Tuple[Any, Any]:
    """Load serialized trained model and preprocessor artifacts."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run 'python src/train.py' first.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preprocessor_path}. Please run 'python src/train.py' first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def predict_single_customer(
    customer_dict: Dict[str, Any],
    model: Any = None,
    preprocessor: Any = None,
) -> Dict[str, Any]:
    """Predict churn probability for a single customer dictionary."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    df_single = pd.DataFrame([customer_dict])
    df_engineered = add_engineered_features(df_single)
    X_transformed = preprocessor.transform(df_engineered)

    churn_prob = float(model.predict_proba(X_transformed)[0, 1])
    churn_pred = int(churn_prob >= 0.5)

    if churn_prob < 0.35:
        risk_level = "Low Risk"
        badge_color = "green"
    elif churn_prob < 0.65:
        risk_level = "Medium Risk"
        badge_color = "orange"
    else:
        risk_level = "High Risk"
        badge_color = "red"

    return {
        "churn_prediction": "Yes" if churn_pred == 1 else "No",
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "badge_color": badge_color,
    }


def predict_batch(
    df: pd.DataFrame, model: Any = None, preprocessor: Any = None
) -> pd.DataFrame:
    """Predict churn probabilities for a batch DataFrame."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    df_out = df.copy()
    df_engineered = add_engineered_features(df)
    X_transformed = preprocessor.transform(df_engineered)

    churn_probs = model.predict_proba(X_transformed)[:, 1]
    churn_preds = (churn_probs >= 0.5).astype(int)

    df_out["Churn_Probability"] = np.round(churn_probs, 4)
    df_out["Predicted_Churn"] = np.where(churn_preds == 1, "Yes", "No")
    df_out["Risk_Level"] = np.where(
        churn_probs < 0.35, "Low Risk", np.where(churn_probs < 0.65, "Medium Risk", "High Risk")
    )

    return df_out
