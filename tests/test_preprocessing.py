"""
Unit Tests for Data Loader & Preprocessing Pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import prepare_features_target, validate_schema
from src.preprocessing import add_engineered_features, create_preprocessor


@pytest.fixture
def sample_raw_dataframe():
    """Fixture providing a sample raw customer DataFrame."""
    return pd.DataFrame(
        [
            {
                "CustomerID": "CUST-001",
                "Gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure": 12,
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.50,
                "TotalCharges": 1026.00,
                "Churn": "Yes",
            },
            {
                "CustomerID": "CUST-002",
                "Gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "Tenure": 36,
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 55.00,
                "TotalCharges": 1980.00,
                "Churn": "No",
            },
        ]
    )


def test_add_engineered_features(sample_raw_dataframe):
    """Test feature engineering logic."""
    df_feat = add_engineered_features(sample_raw_dataframe)

    assert "tenure_years" in df_feat.columns
    assert "charge_ratio" in df_feat.columns
    assert "service_count" in df_feat.columns

    # First customer has OnlineBackup(1) + StreamingTV(1) + StreamingMovies(1) = 3
    assert df_feat.loc[0, "service_count"] == 3
    # Second customer has Security(1) + Backup(1) + DeviceProt(1) + TechSupport(1) = 4
    assert df_feat.loc[1, "service_count"] == 4

    assert np.isclose(df_feat.loc[0, "tenure_years"], 1.0)


def test_prepare_features_target(sample_raw_dataframe):
    """Test schema validation and target transformation."""
    df_feat = add_engineered_features(sample_raw_dataframe)
    X, y = prepare_features_target(df_feat)

    assert len(X) == 2
    assert len(y) == 2
    assert y.iloc[0] == 1  # "Yes" -> 1
    assert y.iloc[1] == 0  # "No" -> 0


def test_preprocessor_transform(sample_raw_dataframe):
    """Test ColumnTransformer fits and transforms without errors."""
    df_feat = add_engineered_features(sample_raw_dataframe)
    X, y = prepare_features_target(df_feat)

    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == 2
    assert X_trans.shape[1] > 10  # Transformed features count
