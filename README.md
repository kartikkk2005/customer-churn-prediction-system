# Customer Churn Prediction System ⚡

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end machine learning pipeline built to predict customer churn from historical customer data, covering data preprocessing, domain feature engineering, cross-validated hyperparameter tuning across multiple classifiers (**Logistic Regression**, **Random Forest**, and **XGBoost**), and an interactive **Streamlit** dashboard for real-time risk prediction and business retention reporting.

---

## 📌 Features & Key Highlights

- **End-to-End Machine Learning Pipeline**: Standardized ingestion, schema validation, domain feature engineering (`tenure_years`, `charge_ratio`, `service_count`), and data scaling.
- **Model Benchmarking & Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation with `GridSearchCV` to optimize classifiers and eliminate overfitting.
- **Classifiers Comparison**: Evaluated and benchmarked **Logistic Regression**, **Random Forest**, and **XGBoost** on Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
- **Interactive Streamlit Dashboard**:
  - **Real-Time Risk Calculator**: Dynamic profile inputs, churn probability gauge, risk level indicator (🟢 Low, 🟡 Medium, 🔴 High), and automated business retention recommendations.
  - **Benchmark Analytics**: Interactive Plotly ROC curves, metric comparison charts, and confusion matrix visualizers.
  - **Batch Prediction Engine**: Drag-and-drop customer CSV dataset uploader with instant downloadable prediction reports.
- **Production Ready**: Full unit test coverage with `pytest`, clean module separation (`src/`), and serialized model artifacts (`joblib`).

---

## 📊 Benchmark Model Performance Results

Below are the benchmarked performance metrics evaluated on a hold-out test set:

| Model Classifier | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.60%** | **61.90%** | **48.00%** | **0.5403** | **0.8273** | **0.8589** |
| **XGBoost Classifier** | 80.40% | 61.22% | 46.15% | 0.5243 | 0.8229 | 0.8491 |
| **Random Forest Classifier** | 80.60% | 63.51% | 41.96% | 0.5026 | 0.8002 | 0.8375 |

> *Note: Models tuned with 5-Fold Stratified Cross-Validation. Top performing model serialized as `models/best_model.joblib`.*

---

## 🛠 Directory Architecture

```
github_project/
├── .streamlit/
│   └── config.toml             # Sleek dark mode visual theme
├── data/
│   ├── generate_data.py        # Historical customer churn dataset generator
│   └── customer_churn.csv      # 2,500 sample customer records
├── models/
│   ├── best_model.joblib       # Serialized top performing ML classifier
│   ├── preprocessor.joblib     # Serialized StandardScaler & OneHotEncoder
│   └── model_metrics.json      # Comparison performance benchmarks
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & schema validator
│   ├── preprocessing.py        # Feature engineering & ColumnTransformer
│   ├── train.py                # GridSearchCV & CV model training engine
│   ├── evaluate.py             # Classification evaluation metric suite
│   └── predict.py              # Single & batch inference interface
├── tests/
│   ├── test_preprocessing.py   # Pytest suite for feature pipeline
│   └── test_model.py           # Pytest suite for inference engine
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git exclusion rules
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-system.git
cd customer-churn-prediction-system

# Install Python requirements
pip install -r requirements.txt
```

### 2. Generate Dataset & Train ML Models

To generate the historical customer dataset and run hyperparameter tuning across all classifiers:

```bash
# Generate synthetic dataset
python data/generate_data.py

# Train models & benchmark performance
python src/train.py
```

### 3. Run Automated Tests

Execute the pytest suite to verify preprocessing and inference logic:

```bash
pytest tests/
```

### 4. Launch Streamlit Web Dashboard

Start the interactive dashboard locally:

```bash
streamlit run app.py
```

The web application will open automatically at `http://localhost:8501`.

---

## 🐙 Adding to GitHub

To push this project to your GitHub account:

1. **Create a new repository** on GitHub named `customer-churn-prediction-system`.
2. **Link local repository and push**:

```bash
# Initialize local Git repository (if not already done)
git init
git add .
git commit -m "feat: Initial commit of Customer Churn Prediction System with ML pipeline & Streamlit dashboard"

# Link to your GitHub remote repository
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction-system.git

# Push code to GitHub
git push -u origin main
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
