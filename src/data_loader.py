"""
Data Loading & Validation Module for Customer Churn Prediction.
"""

import os
from typing import Tuple
import pandas as pd

REQUIRED_COLUMNS = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]

TARGET_COLUMN = "Churn"


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Convert TotalCharges to numeric if read as string
    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == "object":
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["MonthlyCharges"] * df["Tenure"], inplace=True)

    return df


def validate_schema(df: pd.DataFrame, is_training: bool = True) -> bool:
    """Validate DataFrame contains all expected feature columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    if is_training and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from training dataset.")

    return True


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features DataFrame (X) and target binary Series (y)."""
    validate_schema(df, is_training=True)

    drop_cols = [c for c in ["CustomerID", TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    # Normalize target column to 1 (Yes) and 0 (No)
    target_str = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    y = target_str.isin(["yes", "1", "true"]).astype(int)

    return X, y
