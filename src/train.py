"""
Model Training, Benchmarking & Hyperparameter Tuning Pipeline.
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
"""

import os
import sys
import json
from typing import Dict, Any
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_data, prepare_features_target
from src.preprocessing import add_engineered_features, create_preprocessor
from src.evaluate import evaluate_model


def train_and_benchmark(data_path: str = "data/customer_churn.csv", output_dir: str = "models") -> Dict[str, Any]:
    """Train, benchmark, tune hyperparameters, and save best churn prediction models."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw dataset from {data_path}...")
    df_raw = load_data(data_path)
    df_feat = add_engineered_features(df_raw)
    X, y = prepare_features_target(df_feat)

    print(f"Dataset shape: {X.shape}, Target Churn rate: {y.mean():.2%}")

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit feature preprocessor
    print("Fitting feature preprocessor (StandardScaler + OneHotEncoder)...")
    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save preprocessor artifact
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    # Model definitions & Hyperparameter Grids
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
            },
        },
    }

    results = {}
    best_overall_score = -1.0
    best_overall_name = None
    best_overall_model = None

    print("\n" + "=" * 60)
    print("Starting Model Training, Benchmarking & Hyperparameter Tuning...")
    print("=" * 60)

    for name, config in model_configs.items():
        print(f"\n[+] Tuning {name} with GridSearchCV (5-Fold CV)...")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_search.fit(X_train_trans, y_train)

        best_estimator = grid_search.best_estimator_
        print(f"    Best Params: {grid_search.best_params_}")
        print(f"    Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Test set evaluation
        y_pred = best_estimator.predict(X_test_trans)
        y_prob = best_estimator.predict_proba(X_test_trans)[:, 1]

        eval_metrics = evaluate_model(y_test.values, y_pred, y_prob)
        eval_metrics["best_params"] = grid_search.best_params_
        eval_metrics["cv_roc_auc"] = round(float(grid_search.best_score_), 4)

        results[name] = eval_metrics

        print(f"    Test Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"    Test F1-Score: {eval_metrics['f1_score']:.4f}")
        print(f"    Test ROC-AUC:  {eval_metrics['roc_auc']:.4f}")

        # Track overall best model by ROC-AUC
        if eval_metrics["roc_auc"] > best_overall_score:
            best_overall_score = eval_metrics["roc_auc"]
            best_overall_name = name
            best_overall_model = best_estimator

    # Save best overall model
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_overall_model, best_model_path)
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_overall_name} (ROC-AUC = {best_overall_score:.4f})")
    print(f"Saved best model artifact to: {best_model_path}")
    print("=" * 60)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model_name": best_overall_name,
                "best_model_roc_auc": best_overall_score,
                "models": results,
            },
            f,
            indent=2,
        )
    print(f"Saved evaluation metrics to: {metrics_path}")

    return results


if __name__ == "__main__":
    train_and_benchmark()
