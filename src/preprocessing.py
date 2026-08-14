"""
Preprocessing & Feature Engineering Pipeline.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "tenure_years",
    "charge_ratio",
    "service_count",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer business features from raw customer features."""
    df_feat = df.copy()

    # Feature 1: Tenure in years
    df_feat["tenure_years"] = df_feat["Tenure"] / 12.0

    # Feature 2: Monthly vs Total Charge Ratio
    denom = np.maximum(df_feat["TotalCharges"], 1e-5)
    df_feat["charge_ratio"] = df_feat["MonthlyCharges"] / denom

    # Feature 3: Total active optional tech/security services count
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df_feat["service_count"] = 0
    for col in service_cols:
        if col in df_feat.columns:
            df_feat["service_count"] += (df_feat[col] == "Yes").astype(int)

    # Cast SeniorCitizen as object so it is encoded cleanly
    if "SeniorCitizen" in df_feat.columns:
        df_feat["SeniorCitizen"] = df_feat["SeniorCitizen"].astype(str)

    return df_feat


def create_preprocessor() -> ColumnTransformer:
    """Construct a Scikit-Learn ColumnTransformer pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
