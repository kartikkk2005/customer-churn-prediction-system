"""
Unit Tests for Model Prediction & Inference Pipeline.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.predict import predict_single_customer, predict_batch, load_artifacts


@pytest.mark.skipif(
    not os.path.exists("models/best_model.joblib"),
    reason="Trained model artifacts required for inference test",
)
def test_inference_pipeline():
    """Test model loading and prediction functions."""
    model, preprocessor = load_artifacts()

    customer_dict = {
        "Gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 2,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 180.0,
    }

    res = predict_single_customer(customer_dict, model, preprocessor)

    assert "churn_prediction" in res
    assert "churn_probability" in res
    assert "risk_level" in res
    assert 0.0 <= res["churn_probability"] <= 1.0


@pytest.mark.skipif(
    not os.path.exists("models/best_model.joblib"),
    reason="Trained model artifacts required for batch inference test",
)
def test_batch_prediction():
    """Test batch prediction function."""
    df_sample = load_data("data/customer_churn.csv").head(10)
    model, preprocessor = load_artifacts()

    df_res = predict_batch(df_sample, model, preprocessor)

    assert "Churn_Probability" in df_res.columns
    assert "Predicted_Churn" in df_res.columns
    assert "Risk_Level" in df_res.columns
    assert len(df_res) == 10
